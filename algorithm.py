import os
import time
import json
import numpy as np
import pandas as pd
from loguru import logger
from typing import Union
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
from pandas.core import algorithms
from pandas.core.sorting import get_group_index
from cube_alarm.utils import load_data, save_data


@dataclass
class Table:
    df: pd.DataFrame
    mini_data_df: pd.DataFrame
    int2count: dict
    int2val: dict

    @classmethod
    def from_df(cls, data_df: pd.DataFrame, token_nan: int, min_anomaly_ratio: float):
        drop_cols, int2count, int2val = [], {}, {}
        for col in data_df.columns:
            non_null_count = data_df[col].count()
            if data_df[col].nunique() in (1, non_null_count):
                drop_cols.append(col)
                continue
            val2count = data_df[col].value_counts().to_dict()
            int2val[col] = {i + token_nan + 1: k for i, k in enumerate(val2count.keys())}
            val2int = {v: k for k, v in int2val[col].items()}
            int2count[col] = {val2int[k]: v for k, v in val2count.items()}

            data_df.loc[:, col] = data_df[col].map(lambda x: val2int.get(x, token_nan)).astype(np.int32)

        data_df.drop(drop_cols, axis=1, inplace=True)
        data_df.reset_index(drop=True, inplace=True)

        sample_num = int(data_df.shape[0] * min_anomaly_ratio)
        mini_data_df = data_df.sample(n=sample_num, replace=False, random_state=2026)

        return cls(df=data_df, mini_data_df=mini_data_df, int2count=int2count, int2val=int2val)
    
    @classmethod
    def from_dir(cls, data_dir: str):
        df = pd.read_csv(os.path.join(data_dir, "df.csv"))
        mini_data_df = pd.read_csv(os.path.join(data_dir, "mini_data_df.csv"))
        int2count = load_data(os.path.join(data_dir, "int2count.json"))
        int2val = load_data(os.path.join(data_dir, "int2val.json"))

        int2count = {col: {int(k): v for k, v in val2count.items()} for col, val2count in int2count.items()}
        int2val = {col: {int(k): v for k, v in val2count.items()} for col, val2count in int2val.items()}

        return cls(df=df, mini_data_df=mini_data_df, int2count=int2count, int2val=int2val)
    
    def save(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.df.to_csv(os.path.join(data_dir, "df.csv"), index=False)
        self.mini_data_df.to_csv(os.path.join(data_dir, "mini_data_df.csv"), index=False)
        save_data(self.int2count, os.path.join(data_dir, "int2count.json"))
        save_data(self.int2val, os.path.join(data_dir, "int2val.json"))


class RMSProp:
    def __init__(self, param: np.ndarray, lr: float, alpha: float=0.9):
        self.lr = lr
        self.alpha = alpha
        self.momentum = None
        self._eps = 1e-8
        self.param = param

    def step(self, grad: np.ndarray):
        if self.momentum is None:
            self.momentum = np.square(grad)
        else:
            self.momentum = self.alpha * self.momentum + (1 - self.alpha) * np.square(grad)
        
        select_idx = (self.param > 0).astype(np.float32)
        self.param -= self.lr * grad * select_idx / np.sqrt(self.momentum + self._eps)


class Prism:
    def __init__(self, data_source: Union[pd.DataFrame, str]):
        self.token_other = 0
        self.token_nan = 2
        self.token_nan_anomaly = 1
        self._delta = 1
        self._eta = 0.1
        self._eps = 1e-5
        self._normal_sample_ratio = 50
        self.col_weight = {}
        self._min_anomaly_ratio = 0.005
        self._threshold_abs_min = 5
        self._threshold_abs_max = 10
        self._non_null_threshold = 0.3
        self._max_anomaly_frequency = 3
        self._min_recall = 0.9
        if isinstance(data_source, pd.DataFrame):
            self.table = Table.from_df(data_source, self.token_nan, self._min_anomaly_ratio)
        else:
            self.table = Table.from_dir(data_source)
    
    def _unique_columns(self, data_df: pd.DataFrame):
        column2attr = {}
        for column in data_df.columns:
            val_count = data_df[column].value_counts().to_dict()
            nan_num = val_count.pop(self.token_nan, 0)
            if not val_count:
                continue
            val_count_list = [(k, v) for k, v in val_count.items()]
            val_count_list.sort(key=lambda x: x[1], reverse=True)
            if (
                nan_num / data_df.shape[0] < self._non_null_threshold and
                val_count_list[0][-1] / (data_df.shape[0] - nan_num) >= self._min_recall and
                list(val_count.values())[0] >= self._threshold_abs_max
            ):
                column2attr[column] = list(val_count.keys())[0]
        return column2attr

    def drop_duplicates(self, data_df: pd.DataFrame):
        def f(vals) -> tuple[np.ndarray, int]:
            labels, shape = algorithms.factorize(vals, size_hint=vals.shape[0])
            return labels.astype("i8", copy=False), len(shape)

        vals = (col.values for _, col in data_df.items())
        labels, shape = map(list, zip(*map(f, vals)))
        ids = get_group_index(labels, tuple(shape), sort=False, xnull=False)

        counter, id2index = defaultdict(int), {}
        for index, id_ in zip(data_df.index, ids):
            if id_ not in id2index:
                id2index[id_] = index
            counter[id2index[id_]] += 1

        indexs, counter = map(list, zip(*counter.items()))
        return data_df.loc[indexs], np.array(counter)
    
    def _sample_normal_data_df(self, anomaly_index: set, sample_ratio: Union[int, float]):
        sample_num = max(int((sample_ratio + 1) * len(anomaly_index)), 1000)
        sample_num = min(sample_num, self.table.df.shape[0])

        normal_idxs = np.random.choice(self.table.df.index, sample_num, replace=False)
        normal_idxs = [i for i in normal_idxs if i not in anomaly_index]
        normal_data_df = self.table.df.loc[normal_idxs, :]
        return normal_data_df
    
    def _get_combination_attr_num(self, columns2attr: dict):
        if len(columns2attr) == 1:
            col, attr = list(columns2attr.items())[0]
            return self.table.int2count.get(col, {}).get(attr, 0)
        data_df = self.table.mini_data_df.loc[:, list(columns2attr.keys())]
        for col, attr in columns2attr.items():
            data_df = data_df[data_df[col] == attr]
        return max(data_df.shape[0], 1) / self._min_anomaly_ratio
            
    @staticmethod
    def _all_permutations(num: int):
        def _dfs(visited: list, results: list):
            if len(visited) == num:
                if sum(visited):
                    results.append(visited.copy())
                return
            for i in range(1, -1, -1):
                visited.append(i)
                _dfs(visited, results)
                visited.pop(-1)
        
        results = []
        _dfs([], results)
        return np.array(results)
    
    @staticmethod
    def _sample(w: np.ndarray, batch_size: int):
        sample = np.random.rand(batch_size, w.shape[-1])
        sample = (sample < w).astype(np.float32)
        not_zero_index = sample.sum(axis=1) > 0
        return sample[not_zero_index]
    
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
        # precision = precision * sample_weight
        precision = np.einsum('lij,i,j->l', temp, w_n, w_p)
        precision = np.round(alarm_prob / (precision + alarm_prob), decimals=4)

        not_nan_index = (np.einsum('ik,jk->ij', w, nan_anomaly) < 100).astype(np.float32)
        w_p = w_p * not_nan_index
        w_p = w_p / (w_p.sum(axis=-1, keepdims=True) + self._eps)  # (B, P)

        temp = (np.einsum('lk,ijk->lij', w, distance_pp) < self._eps).astype(np.float32)
        recall = np.einsum('li,lj,lij->l', w_p, w_p, temp)
        recall[recall > (1 - 2 * self._min_recall * (1 - self._min_recall))] = 1.

        best_idx, best_metric = np.argmax(precision), (0, 0, -100, 0)
        for i in range(w.shape[0]):
            # if w[i].sum() != 3:
            #     continue
            metric = (recall[i] // 0.002, (precision[i] - w[i].sum() * 0.05) // 0.01, w[i].sum(), precision[i])
            # metric = (precision[i] // 0.01, recall[i], -w[i].sum(), precision[i])
            logger.info('Weight: {}\tPrecision: {:.4f}\tRecall: {:.4f}'.format(
                w[i], precision[i], recall[i]
            ))
            if metric > best_metric:
                best_idx, best_metric = i, metric
        # logger.info('best: {}'.format(w[best_idx]))
        return w[best_idx]

    def grad(
        self, w: np.ndarray, w_sign: np.ndarray, col_weight: np.ndarray, distance_np: np.ndarray,
        distance_pp: np.ndarray, w_n: np.ndarray, w_p: np.ndarray, nan_anomaly: np.ndarray
    ):
        """
        Args:
            w: (B, A), float32
            distance_np: (N, P, A), float32
            distance_pp: (P, P, A), float32
            w_n: (N,), float32
            w_p: (P,), float32
            nan_anomaly, float32
        """
        temp = (np.einsum('lk,ijk->lij', w, distance_np) < self._eps).mean(axis=0, dtype=np.float32)
        precision = w_n @ (1 - temp) @ w_p
        grad_precision = -np.einsum('ij,ijk,i,j->k', temp, distance_np, w_n, w_p) - col_weight * (1 - w.mean(axis=0))

        not_nan_index = (np.einsum('ik,jk->ij', w, nan_anomaly) < self._eps).astype(np.float32)
        w_p = w_p * not_nan_index
        w_p = w_p / (w_p.sum(axis=-1, keepdims=True) + self._eps)  # (B, P)

        temp = (np.einsum('lk,ijk->lij', w, distance_pp) >= self._eps).astype(np.float32)
        sign = (np.einsum('lij,lj->li', temp, w_p) > 0.1).astype(np.float32)
        recall = 1 - np.mean(w_p * sign)
        grad_recall = np.einsum('li,lj,li,lij,ijk->k', w_p, w_p, sign, temp, distance_pp) / w.shape[0]

        sparsity = w.mean()
        logger.info('Value\tPrecision: {:.4f}\tRecall: {:.4f}\tSparsity: {}'.format(precision, recall, sparsity))
        logger.info('Grad of Precision:\n{}\nGrad of Recall:\n{}'.format(grad_precision, grad_recall))
        
        grad = grad_precision + 0.5 * grad_recall
        grad = (grad - grad[w_sign].mean()) / (grad[w_sign].std() + self._eps) + self._eta
        logger.info('Normalized grad:\n{}'.format(grad))
        return grad
    
    def optimize(
        self, distance_np: np.ndarray, distance_pp: np.ndarray, w_n: np.ndarray, w_p: np.ndarray,
        nan_anomaly: np.ndarray, col_weight: np.ndarray, alarm_prob: float
    ):
        attribute_num = distance_np.shape[-1]
        w_p_not_nan = w_p[:, np.newaxis] * (1 - nan_anomaly)  # N * A
        all_nan = w_p_not_nan.sum(axis=0) < self._eps
        w_p_not_nan = w_p_not_nan / (w_p_not_nan.sum(axis=0) + self._eps)

        distance_per_attribute = np.einsum('ik,ijk,jk->k', w_p_not_nan, distance_pp, w_p_not_nan)
        print('distance_per_attribute: {}'.format(distance_per_attribute))
        selected = distance_per_attribute < self._delta
        selected[all_nan] = False
        selected = np.argwhere(selected).reshape(-1)
        distance_pp, distance_np = distance_pp[..., selected], distance_np[..., selected]
        nan_anomaly, col_weight = nan_anomaly[..., selected], col_weight[selected]
        logger.info('Selected: {}'.format(selected))

        max_iter, batch_size = 20, 64
        w = np.ones(distance_pp.shape[-1], dtype=np.float32) * 0.1
        optimizer = RMSProp(w, lr=0.2)
        for i in range(max_iter):
            if (optimizer.param > 0).sum() <= 2:
                break
            logger.info('Epoch {:02}/{}'.format(i + 1, max_iter))
            logger.info('Weight: {}'.format(optimizer.param))
            w = self._sample(optimizer.param, batch_size)
            grad = self.grad(w, optimizer.param > 0, col_weight - 1, distance_np, distance_pp, w_n, w_p, nan_anomaly)
            optimizer.step(grad)
            batch_size = max(batch_size // 2, 32)
        
        logger.info('Weight: {}'.format(optimizer.param))
        w = np.zeros(attribute_num)
        if (optimizer.param > 0).sum():
            w_selected = self.optimal_weight(
                optimizer.param, distance_np, distance_pp, w_n, w_p, nan_anomaly, alarm_prob
            )
            w[selected] = w_selected
        return w
    
    def _optimal_attr_combination(self, w: np.ndarray, anomaly_data_df: pd.DataFrame, attr_valid: dict):
        # 根据异常程度选择属性组合
        selected_columns = anomaly_data_df.columns[w > 0].tolist()
        if not selected_columns:
            return {}

        for col in selected_columns:
            if col not in attr_valid:
                continue
            isin = anomaly_data_df[col].isin(attr_valid[col])
            anomaly_data_df.loc[~isin, col] = self.token_nan_anomaly

        anomaly_index, best_indices, best_alarm_prob = anomaly_data_df.index, [], 0
        attr_combinations = anomaly_data_df.groupby(selected_columns).indices
        for attr_comb, indices in attr_combinations.items():
            if not isinstance(attr_comb, tuple):
                attr_comb = (attr_comb,)
            if self.token_nan_anomaly in attr_comb or len(indices) < self._threshold_abs_max:
                continue
            indices = [int(anomaly_index[i]) for i in indices]
            global_count = self._get_combination_attr_num(
                {col: attr for col, attr in zip(selected_columns, attr_comb)}
            )
            alarm_prob = np.round(len(indices) / (global_count + self._eps), decimals=4)
            if alarm_prob < self._min_anomaly_ratio or alarm_prob <= best_alarm_prob:
                continue
            best_indices, best_alarm_prob = indices, alarm_prob

        cur_results = OrderedDict()
        select_anomaly_df = anomaly_data_df.loc[best_indices, :]
        for col in selected_columns:
            counter = select_anomaly_df[col].value_counts().to_dict()
            nan_num = counter.pop(self.token_nan, 0)
            if len(counter) != 1 or 1 - nan_num / select_anomaly_df.shape[0] <= self._non_null_threshold:
                continue
            cur_results[col] = self.table.int2val[col][list(counter.keys())[0]]
        
        if not cur_results:
            return {}
        return {'pred': cur_results, 'indices': best_indices}
    
    def filter_noise(self, data_df: pd.DataFrame):
        raise NotImplementedError

    def root_cause_analysis(self, anomaly_index: list, normal_data_df: pd.DataFrame):
        if not anomaly_index:
            return None
        ori_anomaly_data_df = self.table.df.loc[anomaly_index, :]
        logger.info('original anomaly_data_df: {}'.format(ori_anomaly_data_df.shape))
        anomaly_data_df, attr_valid = self.filter_noise(ori_anomaly_data_df)
        if not anomaly_data_df.shape[0]:
            return None
        logger.info('filtered anomaly_data_df: {}'.format(anomaly_data_df.shape))
        for col in anomaly_data_df.columns:
            attr_set = attr_valid.get(col, set())
            normal_isin = normal_data_df[col].isin(attr_set)
            normal_data_df.loc[~normal_isin, col] = self.token_other
        normal_data_df = normal_data_df.loc[:, anomaly_data_df.columns]
        alarm_prob = anomaly_data_df.shape[0] / normal_data_df.shape[0]
        
        normal_data_df, w_n = self.drop_duplicates(normal_data_df)
        anomaly_data_df, w_p = self.drop_duplicates(anomaly_data_df)
        logger.info('deduplicated anomaly_data_df: {}\tnormal_data_df: {}'.format(
            anomaly_data_df.shape, normal_data_df.shape)
        )

        col_weight = np.array([self.col_weight.get(col, 1) for col in anomaly_data_df.columns], dtype=np.float32)
        logger.info(anomaly_data_df.columns)

        anomaly, normal = anomaly_data_df.values, normal_data_df.values
        nan_anomaly = anomaly == self.token_nan
        anomaly[nan_anomaly] = self.token_nan_anomaly
        nan_anomaly = nan_anomaly.astype(np.float32)

        distance_anomaly2anomaly = np.not_equal(anomaly[:, np.newaxis, :], anomaly).astype(np.float32)
        distance_normal2anomaly = np.not_equal(normal[:, np.newaxis, :], anomaly).astype(np.float32)

        w_p, w_n = (w_p / w_p.sum()).astype(np.float32), (w_n / w_n.sum()).astype(np.float32)
        w = self.optimize(
            distance_normal2anomaly, distance_anomaly2anomaly,
            w_n, w_p, nan_anomaly, col_weight, alarm_prob
        )

        attr_valid_filtered = {}
        for col in anomaly_data_df.columns:
            attrs = anomaly_data_df[col].unique()
            attr_valid_filtered[col] = {attr for attr in attrs if attr in attr_valid.get(col, set())}
        return self._optimal_attr_combination(
            w, ori_anomaly_data_df.loc[:, anomaly_data_df.columns], attr_valid_filtered
        )

    def __call__(self, anomaly_index: list):
        start = time.time()
        normal_data_df = self._sample_normal_data_df(set(anomaly_index), self._normal_sample_ratio)
        # logger.info('sampled normal_data_df: {}'.format(normal_data_df.shape))

        results = []
        for i in range(self._max_anomaly_frequency):
            # print('Iteration {}/{}'.format(i + 1, self._max_anomaly_frequency))
            cur_result = self.root_cause_analysis(anomaly_index, normal_data_df.copy())
            if not cur_result:
                break
            indices = set(cur_result['indices'])
            anomaly_index = [i for i in anomaly_index if i not in indices]

            results.append(cur_result)
            logger.info(json.dumps(cur_result, ensure_ascii=False))
            if len(results) >= self._max_anomaly_frequency:
                break
        logger.info('Time Consumption: {:.4f}s'.format(time.time() - start))
        return results
