from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from informer_model import InformerEncoderRegressor
from ts_data import TimeSeriesWindowDataset, build_windows, read_dataset, set_seed


@dataclass
class RunResult:
    dataset_name: str
    train_loss: float
    val_loss: float
    test_mse: float
    test_mae: float
    test_rmse: float
    plot_path: str


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    return mse, mae, rmse


def train_one_dataset(
    path: Path,
    seq_len: int,
    pred_len: int,
    known_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    d_model: int,
    n_heads: int,
    e_layers: int,
    d_ff: int,
    dropout: float,
    factor: int,
    patience: int,
    device: torch.device,
    seed: int,
) -> RunResult:
    set_seed(seed)
    x_raw, y_raw = read_dataset(path)
    n = x_raw.shape[0]
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)

    if pred_len != 1:
        raise ValueError("当前实现用于逐点预测曲线，pred_len 必须为 1")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(x_raw[:train_end])
    y_scaler.fit(y_raw[:train_end])
    x = x_scaler.transform(x_raw).astype(np.float32)
    y = y_scaler.transform(y_raw).astype(np.float32)

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = build_windows(
        x=x, y=y, seq_len=seq_len, pred_len=pred_len, train_end=train_end, val_end=val_end
    )

    if train_x.shape[0] == 0 or val_x.shape[0] == 0 or test_x.shape[0] == 0:
        raise ValueError(
            f"{path.name} 样本不足，seq_len={seq_len}, pred_len={pred_len} 下无法构建 train/val/test 窗口"
        )

    train_loader = DataLoader(TimeSeriesWindowDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TimeSeriesWindowDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    model = InformerEncoderRegressor(
        c_in=train_x.shape[-1],
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=dropout,
        factor=factor,
        activation="gelu",
        pred_len=pred_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    last_train_loss = float("inf")
    last_val_loss = float("inf")

    for _epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        last_train_loss = float(np.mean(train_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        last_val_loss = float(np.mean(val_losses))

        if last_val_loss < best_val - 1e-8:
            best_val = last_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    start_idx = max(known_len, seq_len - 1)
    if n <= start_idx:
        raise ValueError(f"known_len/seq_len 过大，无法从 {start_idx + 1} 开始预测（n={n}）")

    infer_x = np.stack([x[t - seq_len + 1 : t + 1] for t in range(start_idx, n)], axis=0)
    infer_loader = DataLoader(
        TimeSeriesWindowDataset(infer_x, np.zeros((infer_x.shape[0], 1), dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    preds = []
    with torch.no_grad():
        for xb, _ in infer_loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)

    y_pred_scaled = np.concatenate(preds, axis=0).reshape(-1, 1)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = y_raw.reshape(-1)[start_idx:]

    mse, mae, rmse = evaluate_regression(y_true, y_pred)

    y_true_full = y_raw.reshape(-1)
    y_pred_full = np.full((n,), np.nan, dtype=np.float32)
    y_pred_full[start_idx:] = y_pred.astype(np.float32)

    fig = plt.figure(figsize=(10, 5))
    x_axis = np.arange(1, n + 1)
    plt.plot(x_axis, y_true_full, label="True Values", linewidth=1.5)
    plt.plot(x_axis, y_pred_full, label="Predictions", linewidth=1.0)
    plt.xlabel("Sample Number")
    plt.ylabel("Target")
    plt.title(path.stem)
    plt.legend()
    fig.tight_layout()
    out_path = path.with_name(f"{path.stem}_pred_vs_true.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return RunResult(
        dataset_name=path.name,
        train_loss=last_train_loss,
        val_loss=last_val_loss,
        test_mse=mse,
        test_mae=mae,
        test_rmse=rmse,
        plot_path=str(out_path),
    )
