import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader

CSV_PATH = "./covtype.csv"      # adjust if you moved the file
BATCH_SIZE =  64                      # tweak to fit your GPU / RAM
VAL_FRAC   =  0.10
TEST_FRAC  =  0.10
RANDOM_SEED =  42

# ------------------------------------------------------------------
# 0)  Load ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

X = df.copy()
X["Distance_To_Hydrology"] = ( (X["Horizontal_Distance_To_Hydrology"] ** 2) + (X["Vertical_Distance_To_Hydrology"] ** 2) ) ** (0.5)
X.drop(["Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology"], axis=1, inplace=True)
# X['Cover_Type'].replace({1:'Spruce/Fir', 2:'Lodgepole Pine', 3:'Ponderosa Pine', 4:'Cottonwood/Willow', 5:'Aspen', 6:'Douglas-fir', 7:'Krummholz'}, inplace=True)
# #We use pandas's 'get_dummies()' method
# X = pd.get_dummies(X)
X = df.drop(columns=["Cover_Type"]).values          # (N, 54)
y = df["Cover_Type"].values.astype(np.int64) - 1    # make labels 0-based

# ------------------------------------------------------------------
# 1)  Scale every feature to [0,1] ---------------------------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X).astype(np.float32)

# ------------------------------------------------------------------
# 2)  Train / val / test split (stratified) ------------------------
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=VAL_FRAC + TEST_FRAC,
    stratify=y, random_state=RANDOM_SEED)

rel_val_frac = VAL_FRAC / (VAL_FRAC + TEST_FRAC)     # split the tmp chunk
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=1 - rel_val_frac,
    stratify=y_tmp, random_state=RANDOM_SEED)

# ------------------------------------------------------------------
# 3)  Torch dataset / dataloader  ----------------------------------
class CovTypeDataset(Dataset):
    """
    Returns (idx, x, y) so that Autoencoder_clust.py can unpack
    `for idx, x, _ in train_loader: …`
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)          # float32 tensor
        self.y = torch.from_numpy(y)          # int64   tensor

    def __len__(self):  return self.X.shape[0]

    def __getitem__(self, idx):
        return idx, self.X[idx], self.y[idx]  # <- triple!

train_ds = CovTypeDataset(X_train, y_train)
val_ds   = CovTypeDataset(X_val,   y_val)
test_ds  = CovTypeDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)
