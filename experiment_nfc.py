import numpy as np
import pandas as pd
from loguru import logger
from algorithm import CubeAlarm
from collections import defaultdict
from utils import load_data


class CubeAlarmNFC(CubeAlarm):
    def __init__(self, data_source: pd.DataFrame | str):
        super().__init__(data_source)
        self._confidence_threshold = 0.9
        self._threshold_rel = 3
        self._max_anomaly_frequency = 4
        self._select_columns = self.table.df.columns[:3]

    def partition(self, data_df: pd.DataFrame, attr_valid: dict, alarm_prob: dict):
        threshold, queue = self._threshold_abs_max, []
        for col in self._select_columns:
            if col not in data_df.columns:
                continue
            counter = data_df[col].value_counts().to_dict()
            counter = [
                [col, k, v, np.round(alarm_prob.get(col, {}).get(k, 0), 4)] for k, v in counter.items()
                if v >= threshold and col in attr_valid and k in attr_valid[col]
            ]
            if not counter:
                continue
            counter.sort(key=lambda x: x[-1])
            if len(counter) > 1 and counter[-1][-1] < self._min_anomaly_ratio:
                counter.sort(key=lambda x: x[-2])
                if len(counter) > 1 and counter[-1][-2] / counter[-2][-2] < 3:
                    continue
            queue.append(counter[-1])

        if not queue:
            return []
        median = np.percentile([x[-2] for x in queue], 50)
        for x in queue:
            x[-2] //= 3 * median
        queue.sort(key=lambda x: x[-2:], reverse=True)

        for col, attr, _, _ in queue:
            select_indices = data_df.index[data_df[col] == attr].tolist()
            select_attrs = self._unique_columns(data_df.loc[select_indices])
            normal_isin = min(
                self._get_combination_attr_num(select_attrs),
                self.table.int2count[col][attr]
            ) - len(select_indices)
            precision = 1 - normal_isin / (self.table.df.shape[0] - data_df.shape[0])
            if (
                 precision < self._confidence_threshold or
                (len(select_indices) + normal_isin) * self._min_anomaly_ratio > len(select_indices)
            ):
                continue
            # logger.info('Select column: {}, attr: {}, num: {}, precision: {:.4f}'.format(
            #     col, self.table.int2val[col][attr], len(select_indices), precision
            # ))
            return select_indices
        return []

    def filter_noise_combination(self, anomaly_data_df: pd.DataFrame, alarm_prob: float, width: int = 2):
        column2int = {col: i for i, col in enumerate(anomaly_data_df.columns)}
        def _get_all_combinations(depth: int):
            if depth == 1:
                return [[col] for col in anomaly_data_df.columns]
            results = []
            for comb in _get_all_combinations(depth - 1):
                tai = column2int[comb[-1]]
                for i in range(tai + 1, len(anomaly_data_df.columns)):
                    results.append(comb + [anomaly_data_df.columns[i]])
            return results
        
        attr_valid = defaultdict(set)
        flags = pd.Series(data=np.zeros(anomaly_data_df.shape[0]), index=anomaly_data_df.index, dtype=bool)
        if alarm_prob >= 1:
            return anomaly_data_df[flags], attr_valid

        anomaly_index = anomaly_data_df.index
        for comb in _get_all_combinations(width):
            max_alarm_ratio, max_index, max_attrs = -1, [], {}
            val_counts = anomaly_data_df.groupby(comb).indices
            for attrs, index in val_counts.items():
                if self.token_nan in attrs or len(index) < self._threshold_abs_max:
                    continue
                index = [anomaly_index[i] for i in index]
                unique_attrs = self._unique_columns(anomaly_data_df.loc[index])
                global_count = self._get_combination_attr_num(unique_attrs)
                alarm_ratio = len(index) / (global_count * alarm_prob)
                if alarm_ratio > max_alarm_ratio:
                    max_alarm_ratio, max_index, max_attrs = alarm_ratio, index, unique_attrs
            if max_alarm_ratio >= self._threshold_rel:
                flags.loc[max_index] = True
                for col, attr in max_attrs.items():
                    attr_valid[col].add(attr)
                max_attrs = {col: self.table.int2val[col][attr] for col, attr in max_attrs.items()}
                # logger.info('Columns&Attributes: {}, Num: {}, Alarm Ratio: {:.4f}'.format(
                #     max_attrs, len(max_index), max_alarm_ratio
                # ))
        
        return anomaly_data_df[flags], attr_valid

    def filter_noise(self, anomaly_data_df: pd.DataFrame):
        def _is_attr_valid(column: str, attr: int, attr_valid: dict):
            index_select = anomaly_data_df[column] == attr
            data_df_select = anomaly_data_df[index_select]
            
            cur_attr_valid = self._unique_columns(data_df_select)
            for col, temp_attr in cur_attr_valid.items():
                attr_valid[col].add(temp_attr)

            normal_isin = self.table.int2count[column][attr] - data_df_select.shape[0]
            normal_num_non_null = self.table.df.shape[0] - anomaly_data_df.shape[0]
            if 1 - normal_isin / normal_num_non_null > self._confidence_threshold:
                return index_select
            return None

        # Placeholder for a new noise filtering method
        attr_valid, alarm_probs = defaultdict(set), defaultdict(dict)
        index = pd.Series(data=np.zeros(anomaly_data_df.shape[0]), index=anomaly_data_df.index, dtype=bool)
        for col in anomaly_data_df.columns:
            val_counts = anomaly_data_df[col].value_counts().to_dict()
            val_counts = {k: v for k, v in val_counts.items() if k != self.token_nan and v >= self._threshold_abs_min}
            if not val_counts:
                continue

            attrs, local_counts = map(np.array, zip(*val_counts.items()))
            global_counts = np.array([self.table.int2count[col].get(k, 0) for k in attrs])
            anomaly_ratio = local_counts / local_counts.sum() / (global_counts / global_counts.sum())
            anomaly_ratio[local_counts < self._threshold_abs_max] = 0

            alarm_prob = local_counts / global_counts
            # for i, attr in enumerate(attrs):
            #     attr_val = self.table.int2val[col][attr]
            #     logger.info('col: {}, attr: {}|{}, num: {}, anomaly_ratio: {:.4f}, alarm_prob: {:.4f}'.format(
            #         col, attr_val, attr, local_counts[i], anomaly_ratio[i], alarm_prob[i]
            #     ))
            alarm_prob[local_counts < self._threshold_abs_max] = 0
            for cur_attr, cur_prob in zip(attrs, alarm_prob):
                if cur_prob:
                    alarm_probs[col][cur_attr] = cur_prob

            condition = (
                (anomaly_ratio >= self._threshold_rel) |
                (local_counts / anomaly_data_df.shape[0] >= 0.8) |
                (alarm_prob >= self._min_anomaly_ratio * 2)
            )
            if alarm_prob.shape[0] == 1:
                condition = condition | (alarm_prob >= self._min_anomaly_ratio)
            select_pos = np.argwhere(condition).reshape(-1)
            for pos in select_pos:
                index_select = _is_attr_valid(col, attrs[pos], attr_valid)
                if index_select is not None:
                    index = index | index_select

        if index.sum():
            anomaly_data_df = anomaly_data_df[index]
        else:
            prob = max([max(v.values()) for v in alarm_probs.values()]) if alarm_probs else 0
            prob = max(prob, self._min_anomaly_ratio)
            logger.info('alarm_prob: {:.4f}'.format(prob))
            for width in range(2, 3):
                filter_anomaly_df, attr_valid = self.filter_noise_combination(
                    anomaly_data_df, prob, width
                )
                if filter_anomaly_df.shape[0]:
                    anomaly_data_df = filter_anomaly_df
                    break

        select_indices = self.partition(anomaly_data_df, attr_valid, alarm_probs)
        anomaly_data_df = anomaly_data_df.loc[select_indices, :]
        
        attr_valid = self._unique_columns(anomaly_data_df)
        attr_valid = {k: {v} for k, v in attr_valid.items()}
        select_columns = [col for col in anomaly_data_df.columns if col in attr_valid]
        anomaly_data_df = anomaly_data_df.loc[:, select_columns]

        return anomaly_data_df, attr_valid


def compute_metrics(preds: list, labels: list):
    """计算 precision, recall, F1.

    Args:
        preds: 预测结果列表，每个元素是 {'pred': {col: value}}
        labels: 标签列表，每个元素是 [{col: value}]
    """
    # 将 pred 转换为集合格式
    pred_set = set()
    for pred in preds:
        if 'pred' not in pred:
            continue
        # 将预测的属性组合转换为字符串格式
        pred_str = '&'.join([f'{k}={v}' for k, v in pred['pred'].items()])
        if pred_str:
            pred_set.add(pred_str)

    # 将 label 转换为集合格式
    label_set = set()
    for label in labels:
        if not label:
            continue
        # label 是 [{col: value}, ...] 格式，取第一个元素
        label_dict = label[0] if isinstance(label, list) else label
        label_str = '&'.join([f'{k}={v}' for k, v in label_dict.items()])
        if label_str:
            label_set.add(label_str)

    # 计算 TP, FP, FN
    tp = len(label_set.intersection(pred_set))
    fp = len(pred_set) - tp
    fn = len(label_set) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == '__main__':
    logger.add('./logs/cube_alarm_nfc.log', mode='w')
    data_path = 'data/nfc/df.csv'
    anomaly_data_list = load_data('./data/label.json')

    model = CubeAlarmNFC(data_path)

    results = defaultdict(list)
    sample_metrics = []  # 保存每个样本的指标

    start_time = time.time()
    for idx, anomaly_data in enumerate(anomaly_data_list):
        logger.info(f'Processing {idx + 1}/{len(anomaly_data_list)}')

        # 执行根因分析
        preds = model(anomaly_data['anomalyIndex'])

        # 计算当前样本的指标
        metrics = compute_metrics(preds, anomaly_data['label'])

        # 保存样本信息
        sample_metrics.append({
            'idx': idx,
            'preds': preds,
            'label': anomaly_data['label'],
            'metrics': metrics
        })

        for k, v in metrics.items():
            results[k].append(v)

        logger.info(f'Sample {idx + 1}: precision={metrics["precision"]:.4f}, recall={metrics["recall"]:.4f}, f1={metrics["f1"]:.4f}')

    # 计算总体指标
    df = pd.DataFrame.from_dict(results)

    tp_total = df['tp'].sum()
    fp_total = df['fp'].sum()
    fn_total = df['fn'].sum()

    final_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    final_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    final_f1 = 2 * tp_total / (2 * tp_total + fp_total + fn_total) if (2 * tp_total + fp_total + fn_total) > 0 else 0

    print('=' * 50)
    print(f'Total samples: {len(anomaly_data_list)}')
    print(f'Total TP: {tp_total}, FP: {fp_total}, FN: {fn_total}')
    print(f'Final Precision: {final_precision:.4f}')
    print(f'Final Recall: {final_recall:.4f}')
    print(f'Final F1: {final_f1:.4f}')
    print('=' * 50)
    print(f'Average Time per Sample: {(time.time() - start_time) / len(anomaly_data_list):.4f} seconds.')
