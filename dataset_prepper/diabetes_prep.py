import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

RAW_PATH = "./diabetes_dataset.csv"
df = pd.read_csv(RAW_PATH)

# Drop the index column that was written by pandas earlier
df = df.drop(columns=["Unnamed: 0"])
cat_cols  = ["Sex", "Ethnicity", "Physical_Activity_Level",
             "Alcohol_Consumption", "Smoking_Status"]
num_cols  = [c for c in df.columns if c not in cat_cols]

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(feature_range=(0, 1)), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

X = preprocessor.fit_transform(df)        # → NumPy float64 array
features = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

class DiabetesDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):  return self.X.shape[0]

    def __getitem__(self, idx):
        # → (index, features, dummy-label) — matches the loops in Autoencoder_clust.py
        return idx, self.X[idx], 0        # label is unused

BATCH_SIZE = 64
torch_ds   = DiabetesDataset(features)
train_loader = DataLoader(torch_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, pin_memory=True)
