import json
import random
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional


def sample_indices(
    data_dir: str = "nfc/",
    sample_size: int = 5,
    filters: Optional[dict[str, str]] = None,
    exclude_label_file: str = "raw_data/nfc_label.json",
    exclude_label_index: Optional[int] = 1
) -> list[int]:
    """
    从 df.csv 中按条件采样指定数量的索引。

    Args:
        data_dir: 数据目录路径
        sample_size: 采样数量
        filters: 列名前缀筛选条件 dict，如 {"customerName": "苏宁"}
        exclude_label_file: 排除的 label 文件路径
        exclude_label_index: label 文件中使用的元素索引

    Returns:
        采样到的行索引列表
    """
    if filters is None:
        filters = {}

    # 1. 加载 int2val.json 获取列值映射
    int2val_path = Path(data_dir) / "int2val.json"
    if not int2val_path.exists():
        raise FileNotFoundError(f"int2val.json not found in {data_dir}")

    with open(int2val_path, 'r', encoding='utf-8') as f:
        int2val = json.load(f)

    # 2. 加载 df.csv
    df_path = Path(data_dir) / "df.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"df.csv not found in {data_dir}")

    df = pd.read_csv(df_path)

    # 3. 如果有筛选条件，按前缀匹配筛选
    if filters:
        mask = pd.Series([True] * len(df))
        for col_name, prefix in filters.items():
            if col_name not in df.columns:
                raise ValueError(f"Column {col_name} not found in df.csv")
            if col_name not in int2val:
                raise ValueError(f"Column {col_name} not found in int2val.json")

            # 获取该列的值映射
            col_val_map = int2val[col_name]  # {int_val: original_str_val}

            # 为每行检查是否匹配前缀
            def matches_prefix(row_idx):
                int_val = str(df.iloc[row_idx][col_name])
                original_val = col_val_map.get(int_val, "")
                return str(original_val).startswith(prefix)

            mask = mask & pd.Series([matches_prefix(i) for i in range(len(df))], index=df.index)

        valid_indices = df.index[mask].tolist()
    else:
        valid_indices = df.index.tolist()

    # 4. 如果指定了 exclude_label_index，排除 anomalyIndex
    if exclude_label_file and exclude_label_index is not None:
        label_path = Path(exclude_label_file)
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)

            if isinstance(labels, list) and len(labels) > exclude_label_index:
                exclude_indices = set(labels[exclude_label_index].get("anomalyIndex", []))
                valid_indices = [idx for idx in valid_indices if idx not in exclude_indices]

    # 5. 随机采样
    if len(valid_indices) <= sample_size:
        return valid_indices

    return random.sample(valid_indices, sample_size)


def plot_line(data_df: pd.DataFrame, vertical_line: Optional[dict]=None):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    data_df.plot(kind='line', ax=ax, grid=True)
    ymin, ymax = ax.get_ylim()
    colors = ('red', 'blue', 'red')
    linestyles = (':', '-.', '--')
    if vertical_line:
        for i, (label, x) in enumerate(vertical_line.items()):
            ax.vlines(
                x=x, ymin=ymin, ymax=ymax, color=colors[i], linestyle=linestyles[i],
                linewidth=2, alpha=0.5, label=label
            )
    ax.legend()
    plt.show()
    # plt.savefig()


def load_data(data_path: str):
    data = []
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as lines:
            for line in lines:
                data.append(json.loads(line.strip()))
    elif data_path.endswith('.yaml'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    return data


def save_data(data, save_path: str, mode='w'):
    if save_path.endswith('.json'):
        with open(save_path, mode, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif save_path.endswith('.jsonl'):
        if not isinstance(data, (list, tuple)):
            data = [data]
        with open(save_path, mode, encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    elif save_path.endswith('.yaml'):
        with open(save_path, mode, encoding='utf-8') as f:
            yaml.safe_dump(data, f)


