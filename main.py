from pathlib import Path

import torch

from train_eval import train_one_dataset

DATA_FILE = "data/副本30v.xlsx"
KNOWN_LEN = 10
SEQ_LEN = 10
PRED_LEN = 1
BATCH_SIZE = 64
EPOCHS = 40
LR = 2e-3
D_MODEL = 128
N_HEADS = 4
E_LAYERS = 1
D_FF = 256
DROPOUT = 0.0
FACTOR = 5
PATIENCE = 20
SEED = 42

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_FILE}")

    res = train_one_dataset(
        path=data_path,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        known_len=KNOWN_LEN,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        factor=FACTOR,
        patience=PATIENCE,
        device=device,
        seed=SEED,
    )

    print(f"\n[{res.dataset_name}]")
    print(f"loss(train_last_epoch)={res.train_loss:.6f}  loss(val_last_epoch)={res.val_loss:.6f}")
    print(f"mse(test)={res.test_mse:.6f}  mae(test)={res.test_mae:.6f}  rmse(test)={res.test_rmse:.6f}")
    print(f"plot_saved={res.plot_path}")


if __name__ == "__main__":
    main()
