# ./prepare_marketing.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class MarketingDataset(Dataset):
    """
    Returns (idx, x, y) where
      idx : int  – sample index   (needed by Autoencoder_clust.py)
      x   : FloatTensor (input_dim,) – scaled feature vector
      y   : LongTensor  (1,)        – optional label (Response)
    """
    def __init__(self, csv_path: str, scaler: MinMaxScaler | None = None):
        # ---------- 1.  Read ----------------------------------------------------------------
        df = pd.read_csv(csv_path, sep="\t")          # dataset is TAB-separated

        # ---------- 2.  Feature engineering -----------------------------------------------
        # ---- Handle dates
        df["Dt_Customer"]  = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
        ref_date           = df["Dt_Customer"].max()
        df["Cust_Tenure"]  = (ref_date - df["Dt_Customer"]).dt.days.astype(int)

        # ---- Age instead of Year_Birth
        df["Age"] = ref_date.year - df["Year_Birth"]

        # ---------- 3.  Drop / keep columns ------------------------------------------------
        df = df.drop(columns=["ID",                       # surrogate key
                              "Year_Birth",
                              "Dt_Customer"])             # original string

        # ---------- 4.  Categorical → one-hot ---------------------------------------------
        cat_cols  = ["Education", "Marital_Status"]
        df        = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # ---------- 5.  Missing values -----------------------------------------------------
        df["Income"] = df["Income"].fillna(df["Income"].median())

        # ---------- 6.  Scale to [0,1] -----------------------------------------------------
        feats = df.drop(columns=["Response"])             # numeric + one-hot
        if scaler is None:
            scaler = MinMaxScaler()
            x = scaler.fit_transform(feats)
        else:
            x = scaler.transform(feats)

        # ---------- 7.  Final tensors ------------------------------------------------------
        self.x      = torch.tensor(x, dtype=torch.float32)
        self.y      = torch.tensor(df["Response"].values, dtype=torch.long)
        self.idx    = torch.arange(len(df))
        self.scaler = scaler
        self.n_feat = self.x.shape[1]                     # ← input_dim to pass into AE

    # -- Required by torch.utils.data.Dataset ----------------------------------------------
    def __len__(self):         return len(self.x)
    def __getitem__(self, i):  return self.idx[i], self.x[i], self.y[i]

# Convenience helper -----------------------------------------------------------------------
def build_loader(csv_path, batch_size=256, shuffle=True):
    ds  = MarketingDataset(csv_path)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return dl, ds.n_feat, ds.scaler
