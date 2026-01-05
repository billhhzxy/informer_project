import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, x_windows: np.ndarray, y_windows: np.ndarray):
        self.x = torch.from_numpy(x_windows).float()
        self.y = torch.from_numpy(y_windows).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_windows(
    x: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    pred_len: int,
    train_end: int,
    val_end: int,
):
    if pred_len != 1:
        raise ValueError("当前实现用于逐点预测曲线，pred_len 必须为 1")

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    n = x.shape[0]
    last_start = n - seq_len + 1
    for i in range(last_start):
        target_idx = i + seq_len - 1
        x_win = x[i : i + seq_len]
        y_win = y[target_idx : target_idx + 1].reshape(-1)
        if target_idx < train_end:
            train_x.append(x_win)
            train_y.append(y_win)
        elif target_idx < val_end:
            val_x.append(x_win)
            val_y.append(y_win)
        else:
            test_x.append(x_win)
            test_y.append(y_win)

    def _stack(a):
        if len(a) == 0:
            return np.empty((0,))
        return np.stack(a, axis=0)

    return (_stack(train_x), _stack(train_y)), (_stack(val_x), _stack(val_y)), (_stack(test_x), _stack(test_y))


def read_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 5:
        raise ValueError(f"{path.name} 需要至少5列数值数据，当前只有 {numeric.shape[1]} 列")

    used = numeric.iloc[:, -5:]
    x = used.iloc[:, :4].to_numpy(dtype=np.float32)
    y = used.iloc[:, 4].to_numpy(dtype=np.float32).reshape(-1, 1)
    return x, y
