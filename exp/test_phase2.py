# test_phase2.py — for log(1+AHI) trained models (LSTM / Transformer)

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import os, sys, argparse
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "utils"))

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
torch.set_num_threads(16)

# ---------------- Models ----------------
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
        return self.fc2(y)

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
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
        nn.init.constant_(self.head.bias, 10.0)  # 초기 예측값을 높게 설정

    def forward(self, x):
        z = self.proj(x)
        z = self.posenc(z)
        cls_tokens = self.cls.expand(x.size(0), 1, -1)
        z = torch.cat([cls_tokens, z], dim=1)
        z = self.encoder(z)
        cls = z[:, 0, :]
        cls = self.norm(cls)
        cls = self.dropout(cls)
        return self.head(cls)

# --------------- Dataset ----------------
def _safe_float(x):
    if isinstance(x, (bytes, bytearray)): x = x.decode('utf-8', errors='ignore')
    if isinstance(x, (list, tuple)) and len(x) == 1: x = x[0]
    try: return float(x)
    except (TypeError, ValueError): return np.nan

class AHIDataset(Dataset):
    """df: ['Patient_ID','Features','Actual_AHI'] — Features: (W,30,D)"""
    def __init__(self, df):
        self.features = df["Features"].values
        self.ahi_values = df["Actual_AHI"].values  # Linear AHI
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        feats = self.features[idx]
        ahi   = _safe_float(self.ahi_values[idx])
        feats_2d = feats.reshape(feats.shape[0], -1)
        return torch.tensor(feats_2d, dtype=torch.float32), torch.tensor(ahi, dtype=torch.float32)

# --------------- Test ----------------
def test_phase2(args):
    ckpt_dir = os.path.join(args.checkpoints, args.model_id)
    folder_path = os.path.join('../test_results', args.model_id, args.dataset_name)

    # Load test H5
    patient_features, patient_ahi, patient_ids = [], [], []
    with h5py.File(os.path.join(folder_path, 'patient_features_test.h5'), 'r') as f:
        for pid in tqdm(list(f.keys()), desc="Loading HDF5"):
            g = f[pid]
            if 'features' not in g or 'Actual_AHI' not in g.attrs: continue
            patient_ids.append(pid)
            patient_features.append(g['features'][:])
            patient_ahi.append(g.attrs['Actual_AHI'])

    df = pd.DataFrame({'Patient_ID': patient_ids, 'Features': patient_features, 'Actual_AHI': patient_ahi}).dropna()
    ds = AHIDataset(df)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = df['Features'].iloc[0]
    input_dim = sample.shape[1] * sample.shape[2]

    if args.model_type == 'lstm':
        model = AHIRegressor(input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        ckpt_name = f'checkpoint_{args.model_id}_phase2_logweighted_lstm.pth'
    else:
        model = AHITransformer(input_dim=input_dim, d_model=args.d_model, nhead=args.nhead,
                               num_layers=args.num_layers, dropout=args.dropout).to(device)
        ckpt_name = f'checkpoint_{args.model_id}_phase2_logweighted_tfm.pth'

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and any(k in state for k in ("state_dict","model","net")):
        state = state.get("state_dict", state.get("model", state.get("net")))
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded model: {ckpt_path}")

    preds_ahi, trues_ahi = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Phase 2 Inference"):
            x = x.to(device)
            yhat_log = model(x).squeeze(1)
            # --- log → linear 복원 ---
            yhat = np.expm1(yhat_log.cpu().numpy())
            y_true = y.view(-1).cpu().numpy()
            preds_ahi.extend(yhat.tolist())
            trues_ahi.extend(y_true.tolist())

    # --- Metrics ---
    rmse = float(np.sqrt(mean_squared_error(trues_ahi, preds_ahi)))
    mae  = float(mean_absolute_error(trues_ahi, preds_ahi))
    r, p = pearsonr(trues_ahi, preds_ahi)
    print(f"\n✅ AHI Regression — RMSE: {rmse:.3f}, MAE: {mae:.3f}, r={r:.2f}, p={p:.1e}")

    # --- Scatter plot ---
    m = max(max(trues_ahi), max(preds_ahi)) if len(trues_ahi) else 1.0
    plt.figure(figsize=(7,6))
    plt.scatter(trues_ahi, preds_ahi, alpha=0.5, color='royalblue')
    plt.plot([0, m], [0, m], 'r--', lw=1)
    plt.xlabel('True AHI'); plt.ylabel('Predicted AHI')
    plt.title(f'True vs Predicted AHI ({args.model_type.upper()})')
    plt.text(5, m*0.9, f"r = {r:.2f}\np = {p:.1e}\nMAE = {mae:.2f}", fontsize=10,
             bbox=dict(facecolor='white', alpha=0.6))
    plt.grid(True); plt.tight_layout(); plt.show()

    # --- Save CSV ---
    out_csv = os.path.join(folder_path, f'ahi_results_{args.model_type}_logweighted.csv')
    pd.DataFrame({
        'Patient_ID': df['Patient_ID'],
        'AHI Target': trues_ahi,
        'AHI Prediction': preds_ahi
    }).to_csv(out_csv, index=False)
    print(f"Saved results: {out_csv}")

    return {"rmse": rmse, "mae": mae, "r": r, "p": p}

# --------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Phase 2 AHI Regression Testing (log-scale model)')
    ap.add_argument('--checkpoints', type=str, default='../checkpoints/')
    ap.add_argument('--model_id', type=str, default='260109_30s_multicenter_C4A1_multitask_128_16_6331157_11975_0.001_wce_revisedlabel_kiss_mesa_mros_internal')
    ap.add_argument('--dataset_name', type=str, default='internal/Internal')
    ap.add_argument('--model_type', choices=['lstm','tfm'], default='tfm')
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=4)
    ap.add_argument('--num_layers', type=int, default=3)

    args = ap.parse_args()
    test_phase2(args)
    print("Phase 2 testing script completed.")