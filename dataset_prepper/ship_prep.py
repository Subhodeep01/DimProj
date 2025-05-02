# ship_dataset_prep.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from pathlib import Path

# -------------------------------------------------
# 1.  LOAD the raw CSV
# -------------------------------------------------
DATA_PATH = Path(__file__).with_name("Ship_Performance_Dataset.csv")
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# 2.  BASIC CLEAN-UP
#     • drop obvious identifiers / free-text columns
#     • keep rows that have at least 80 % of the columns filled
# -------------------------------------------------
ID_LIKE   = ["voyage_id", "ship_id", "IMO", "timestamp"]
TEXT_LIKE = ["remarks", "notes"]
df = df.drop(columns=[c for c in ID_LIKE + TEXT_LIKE if c in df.columns])

row_thresh = int(df.shape[1] * 0.80)
df = df.dropna(thresh=row_thresh).reset_index(drop=True)

# -------------------------------------------------
# 3.  FEATURE–ENGINEERING PIPELINE
#     • numeric  ➜ Min-Max 0-1 (matches sigmoid decoder)
#     • category ➜ One-Hot then Min-Max
# -------------------------------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

ct = ColumnTransformer(
    [
        ("num",  MinMaxScaler(feature_range=(0, 1)), num_cols),
        ("cat",  OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)
features = ct.fit_transform(df)
features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

# -------------------------------------------------
# 4.  TORCH DATASET / DATALOADER
#     Autoencoder expects (idx, x, _) in each mini-batch
#     and will call .to(device) on *both* idx and x
# -------------------------------------------------
class ShipPerfDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # label placeholder is 0 – it is ignored by the model
        return torch.tensor(idx, dtype=torch.long), self.X[idx], torch.tensor(0)

BATCH_SIZE = 64
dataset     = ShipPerfDataset(features)
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=0,     # set >0 if you have spare CPU cores
    drop_last=False,
)

print(f"Prepared dataset with shape {features.shape}")
print(f"Suggested input_dim = {features.shape[1]}")
