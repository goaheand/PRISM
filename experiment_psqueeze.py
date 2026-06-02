import os
import re
import time
import numpy as np
import pandas as pd
from loguru import logger
from algorithm import CubeAlarm
from collections import defaultdict, OrderedDict


class CubeAlarmSqueeze(CubeAlarm):
    def __init__(self, data_source: pd.DataFrame | str):
        super().__init__(data_source)
        self._threshold_abs_max = 5
        self._min_recall = 0.9

    def partition(self, data_df: pd.DataFrame):
        max_col, max_count, threshold = 0, 0, 5
        for col in data_df.columns:
            count = data_df[col].value_counts().max()
            if count > max_count:
                max_col, max_count = col, count
        if max_count < threshold:
            return []
        counter = data_df[max_col].value_counts().to_dict()
        counter_list = [(k, v) for k, v in counter.items() if v >= threshold]
        counter_list.sort(key=lambda x: x[-1])
        if (
            len(counter_list) > 1 and
            counter_list[-1][-1] / counter_list[-2][-1] < 3 and
            data_df.shape[0] > 3 * counter_list[-1][-1]
        ):
            return []

        select_indices = data_df[max_col] == counter_list[-1][0]
        select_data_df = data_df.loc[select_indices, :]

        result = []        
        select_attrs = self._unique_columns(select_data_df)
        normal_isin = self._get_combination_attr_num(select_attrs) - len(select_indices)
        if normal_isin / (self.table.df.shape[0] - data_df.shape[0]) > 1 - self._min_recall:
            select_cols = [col for col in data_df.columns if col not in select_attrs]
            result = self.partition(data_df.loc[:, select_cols])

        if not result:
            result = data_df.index[select_indices].tolist()
        return result

    def filter_noise(self, anomaly_data_df: pd.DataFrame):
        attr_valid = OrderedDict()
        index = self.partition(anomaly_data_df)
        anomaly_data_df = anomaly_data_df.loc[index, :]
        for col in anomaly_data_df.columns:
            val_counts = anomaly_data_df[col].value_counts().to_dict()
            val_counts = {k: v for k, v in val_counts.items() if k != self.token_nan and v >= self._threshold_abs_max}

            for attr, count in val_counts.items():
                attr_val = self.table.int2val[col][attr]
                print('col: {}, attr: {}, num: {}'.format(col, attr_val, count))

            if not (0 < len(val_counts) < 4):
                continue

            attr_valid[col] = list(val_counts.keys())

        return anomaly_data_df.loc[:, list(attr_valid.keys())], attr_valid

    def optimal_weight(
        self, w: np.ndarray, distance_np: np.ndarray, distance_pp: np.ndarray, w_n: np.ndarray,
        w_p: np.ndarray, nan_anomaly: np.ndarray, alarm_prob: float
    ):
        non_zero_index = w > 0
        select_num = non_zero_index.sum()

        w = np.zeros((2 ** select_num - 1, w.shape[-1]), dtype=np.int32)
        w[:, non_zero_index] = self._all_permutations(select_num)

        temp = (np.einsum('lk,ijk->lij', w, distance_np) < self._eps).astype(np.float32)
        # precision = 1 - np.einsum('lij,i,j->l', temp, w_n, w_p)
        precision = np.einsum('lij,i,j->l', temp, w_n, w_p)
        precision = np.round(alarm_prob / (precision + alarm_prob), decimals=4)

        not_nan_index = (np.einsum('ik,jk->ij', w, nan_anomaly) < 100).astype(np.float32)
        w_p = w_p * not_nan_index
        w_p = w_p / (w_p.sum(axis=-1, keepdims=True) + self._eps)  # (B, P)

        temp = (np.einsum('lk,ijk->lij', w, distance_pp) < self._eps).astype(np.float32)
        recall = np.einsum('li,lj,lij->l', w_p, w_p, temp)
        # recall[recall > (1 - 2 * self._min_recall * (1 - self._min_recall))] = 1.

        best_idx, best_metric = np.argmax(precision), (0, 0, -100, 0)
        for i in range(w.shape[0]):
            # if w[i].sum() != 2:
            #     continue
            metric = (recall[i] // 0.002, (precision[i] - w[i].sum() * 0.05) // 0.01, w[i].sum(), precision[i])
            # metric = (precision[i] // 0.01, recall[i], -w[i].sum(), precision[i])
            logger.info('Weight: {}\tPrecision: {:.4f}\tRecall: {:.4f}'.format(w[i], precision[i], recall[i]))
            if metric > best_metric:
                best_idx, best_metric = i, metric
        logger.info('best: {}'.format(w[best_idx]))
        return w[best_idx]


def anomaly_detection(data_df: pd.DataFrame, threshold: float = 0.3, ignore_min_val: float = 5.0):
    def optimal_threshold(data: np.ndarray):
        data = np.sort(data)
        v1 = 1 - data / data[-1]
        v2 = np.ones_like(data) * data.shape[0]
        for i in range(1, data.shape[-1]):
            v2[i] = v2[i - 1] if data[i] == data[i - 1] else data.shape[0] - i
        v2 = v2 / v2.shape[0]
        
        anomaly_ratio = v1 / (v2 + 1e-5)
        optimal_idx = np.argmax(anomaly_ratio)
        return data[optimal_idx]

    mape = ((data_df['predict'] - data_df['real']).abs() / (data_df['real'] + 1e-5)).values
    mae = (data_df['predict'] - data_df['real']).abs().values
    mape[mae < ignore_min_val] = 0.
    threshold = min(optimal_threshold(mape), threshold)
    anomaly_indices = data_df[mape > threshold].index.tolist()

    data_df.drop(['real', 'predict'], axis=1, inplace=True)
    return data_df, anomaly_indices


def f1_score(pred: list, label: list):
    label = set(label)
    pred = {'&'.join([f'{k}={v}' for k, v in x['pred'].items()]) for x in pred}
    tp = len(label.intersection(pred))
    fp = len(pred) - tp
    fn = len(label) - tp
    f1_score = 2 * tp / (2 * tp + fp + fn)
    result = {'tp': tp, 'fp': fp, 'fn': fn, 'f1_score': f1_score}
    return result


if __name__ == '__main__':
    np.random.seed(12)
    # A
    # data_dir = 'data/Psqueeze-dataset/A/A_week_12_cuboid_layer_1_n_ele_3'
    # selected_columns = ['i', 'e', 'c', 'p', 'l']
    # ad_args = {'threshold': 0.3, 'ignore_min_val': 5.0}
    # B
    # data_dir = 'data/Psqueeze-dataset/B/B_cuboid_layer_3_n_ele_3'
    # selected_columns = ['logstream_idc', 'logstream_province', 'logstream_isp', 'cmatch']
    # ad_args = {'threshold': 3, 'ignore_min_val': 5.0}
    # D
    data_dir = 'data/Psqueeze-dataset/D/B_cuboid_layer_1_n_ele_1'
    selected_columns = ['a', 'b', 'c', 'd']
    ad_args = {'threshold': 0.3, 'ignore_min_val': 0}

    label_path = os.path.join(data_dir, 'injection_info.csv')
    log_dir = os.path.join('logs', '/'.join(data_dir.split('/')[-2:]))
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, 'result.log'), mode='w')

    labels = pd.read_csv(label_path)
    labels = {
        int(row['timestamp']): row['set'].split(';')
        for _, row in labels.iterrows()
    }

    start = time.time()
    results = defaultdict(list)
    for file_name in os.listdir(data_dir):
        print('file_name: {}'.format(file_name))
        if not re.match(r'\d+\.csv', file_name):
            continue
        timestamp = int(file_name.split('.csv')[0])
        if timestamp not in labels:
            continue
        # if file_name != '1450681920.csv':
        #     continue
        file_path = os.path.join(data_dir, file_name)
        data_df = pd.read_csv(file_path)
        data_df, anomaly_indices = anomaly_detection(data_df, **ad_args)
        if len(anomaly_indices) < 10:
            continue

        preds = CubeAlarmSqueeze(data_df)(anomaly_indices)

        result = f1_score(preds, labels[timestamp])
        for k, v in result.items():
            results[k].append(v)
        
        logger.info('file_name: {}, f1_score: {:.4f}'.format(file_name, result['f1_score']))

    df = pd.DataFrame.from_dict(results)

    tp, fp, fn = [df[k].sum() for k in ('tp', 'fp', 'fn')]
    final_f1_score = 2 * tp / (2 * tp + fp + fn)
    final_precision = tp / (tp + fp)
    final_recall = tp / (tp + fn)
    print('Average time cost: {:.4f}s.'.format((time.time() - start) / df.shape[0]))
    logger.info('precision: {:.4f}, recall: {:4f}, f1_score: {:.4f}'.format(
        final_precision, final_recall, final_f1_score
    ))
