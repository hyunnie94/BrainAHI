import os, sys, argparse, json
import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import wandb

# ---------------------------
# utils
# ---------------------------
this_dir   = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "utils"))
from utils.losses import weighted_mae_loss
from utils.tools import adjust_learning_rate

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
torch.set_num_threads(16)


def build_criterion(loss_type):
    if loss_type == 'weighted_mae':
        return lambda pred, target: weighted_mae_loss(pred, target)
    else:
        return nn.SmoothL1Loss()

# ========================
# Model
# ========================
class AHIRegressor(nn.Module):
    """Bi-LSTM → 2층 MLP"""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        y, _ = self.lstm(x)
        y = y[:, -1, :]
        y = self.relu(self.fc1(y))
        y = self.dropout(y)
        return self.fc2(y)      # (B,1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:L, :]


class AHITransformer(nn.Module):
    """Transformer Encoder → CLS pooling → FC(회귀)"""
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.posenc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
        nn.init.constant_(self.head.bias, 5.0)   # 초기 예측이 0보다 약간 높게

    def forward(self, x):
        z = self.proj(x)
        z = self.posenc(z)
        cls_token = self.cls.expand(x.size(0), 1, -1)
        z = torch.cat([cls_token, z], dim=1)
        z = self.encoder(z)
        cls = z[:, 0, :]
        cls = self.norm(cls)
        cls = self.dropout(cls)
        return self.head(cls)

# ========================
# Dataset  (로그 스케일)
# ========================
class AHIDataset(Dataset):
    """Features: (N, 30, D), Target: log(1+AHI)"""
    def __init__(self, df):
        self.X = df["Features"].values
        self.y = np.log1p(df["Actual_AHI"].values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feats = self.X[idx]
        ahi_log = float(self.y[idx])
        feats_2d = feats.reshape(feats.shape[0], -1)
        return torch.tensor(feats_2d, dtype=torch.float32), torch.tensor(ahi_log, dtype=torch.float32)

# ========================
# H5 → DataFrame loader
# ========================
def load_h5_to_df(h5_path, min_windows=1, verbose=True):
    rows = []
    with h5py.File(h5_path, "r") as f:
        for pid in tqdm(list(f.keys()), desc="Scan H5"):
            grp = f[pid]
            if 'features' not in grp or 'Actual_AHI' not in grp.attrs:
                continue
            feats = grp['features'][:]
            if feats.ndim != 3 or feats.shape[0] < min_windows:
                continue
            ahi = float(grp.attrs['Actual_AHI'])
            if np.isnan(ahi): continue
            rows.append({"Patient_ID": pid, "Features": feats, "Actual_AHI": ahi})
    df = pd.DataFrame(rows)
    if verbose and len(df):
        print("[H5] Loaded:", len(df), "patients")
        print("[H5] AHI range:", df["Actual_AHI"].min(), "-", df["Actual_AHI"].max())
    return df

# ========================
# Train loop
# ========================
def train_phase2(args):
    wandb.init(
        project="BrainAHI_Multitask_Phase2",
        name=f"phase2_train_logweighted_{args.model_id}",
        config=vars(args),
    )

    path = os.path.join(args.checkpoints, args.model_id)
    h5_path = os.path.join(path, "patient_features_phase1_train.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    df = load_h5_to_df(h5_path, min_windows=args.min_windows)
    if len(df) == 0:
        raise RuntimeError("No valid training samples")

    full_ds = AHIDataset(df)
    if args.val_ratio > 0:
        n_total = len(full_ds)
        n_val = max(1, int(n_total * args.val_ratio))
        n_tr = n_total - n_val
        tr_ds, val_ds = random_split(full_ds, [n_tr, n_val],
                                     generator=torch.Generator().manual_seed(42))
    else:
        tr_ds, val_ds = full_ds, None

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4) if val_ds else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = df['Features'].iloc[0]
    input_dim = sample.shape[1] * sample.shape[2]

    if args.model_type == "lstm":
        model = AHIRegressor(input_dim=input_dim, hidden_dim=args.hidden_dim,
                             dropout=args.dropout).to(device)
    else:
        model = AHITransformer(input_dim=input_dim, d_model=args.d_model,
                               nhead=args.nhead, num_layers=args.num_layers,
                               dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, steps_per_epoch=len(tr_loader),
        pct_start=args.pct_start, epochs=args.train_epochs,
        max_lr=args.learning_rate
    )
    criterion = build_criterion(args.loss_type)

    best_val = np.inf
    best_path = os.path.join(path, f'checkpoint_{args.model_id}_phase2_logweighted_tfm.pth')

    for ep in range(1, args.train_epochs + 1):
        model.train()
        tr_losses = []
        for x, y in tqdm(tr_loader, desc=f"Train ep{ep}/{args.train_epochs}"):
            x, y = x.to(device), y.view(-1).to(device)
            optimizer.zero_grad()
            yhat = model(x).squeeze(1)
            loss = criterion(yhat, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            tr_losses.append(loss.item())
        tr_loss = np.mean(tr_losses)

        logs = {"epoch": ep, "train_loss": tr_loss}
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.view(-1).to(device)
                    yhat = model(x).squeeze(1)
                    val_losses.append(criterion(yhat, y).item())
            val_loss = np.mean(val_losses)
            logs["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), best_path)

        scheduler.step()
        wandb.log(logs)
        print(f"[Epoch {ep}] {json.dumps(logs)}")

    if val_loader is None:
        torch.save(model.state_dict(), best_path)
    print(f"✅ Phase 2 model saved at: {best_path}")
    return model

# ========================
# CLI
# ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', type=str, default='../checkpoints/')
    ap.add_argument('--model_id', type=str, default='260109_30s_multicenter_C4A1_multitask_128_16_6331157_11975_0.001_wce_revisedlabel_kiss_mesa_mros_internal')
    ap.add_argument('--train_epochs', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--learning_rate', type=float, default=5e-4)
    ap.add_argument('--pct_start', type=float, default=0.1)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--min_windows', type=int, default=1)
    ap.add_argument('--model_type', choices=['lstm', 'tfm'], default='tfm')
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--loss_type', choices=['weighted_mae'], default='weighted_mae')
    args = ap.parse_args()
    train_phase2(args)
    print("Phase 2 training script completed.")

if __name__ == "__main__":
    main()