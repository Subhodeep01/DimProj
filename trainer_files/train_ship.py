# --- prerequisites -----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Autoencoder_clust_fixed import Autoencoder               # same folder
from ship_prep import train_loader, features               # code above
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder(
        cluster_num = 4,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 1e-3,
        input_dim   = features.shape[1],           # ← **critical change**
        hidden_dim  = 64,
        bottleneck_dim = 3
).to(device)

PRE_EPOCHS   = 100
FINET_EPOCHS = 500
UPDATE_EVERY = 10

# 1) unsupervised reconstruction pre-training
model.pretrain(train_loader,
               batch_size=train_loader.batch_size,
               pretrain_epoch=PRE_EPOCHS)
               
torch.save(model.state_dict(), "./model_ship_pretrain.pt")

# 2) self-training + clustering fine-tuning
pred_labels = model.finetrain(train_loader,
                              batch_size=train_loader.batch_size,
                              train_epoch=FINET_EPOCHS,
                              update_epoch=UPDATE_EVERY)

# Specify the path to save the model
PATH = "./model_ship.pt"

# Save the model's state_dict
torch.save(model.state_dict(), PATH)