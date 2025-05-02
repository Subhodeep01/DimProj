# --- prerequisites -----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Autoencoder_clust import Autoencoder               # same folder
from covtype_prep import train_loader               # code above
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder(
        cluster_num = 7,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 1e-4,
        input_dim   = 54,           # ← **critical change**
        hidden_dim  = 32,
        bottleneck_dim = 3
).to(device)

PRE_EPOCHS   = 10
FINET_EPOCHS = 100
UPDATE_EVERY = 5

# 1) unsupervised reconstruction pre-training
model.pretrain(train_loader,
               batch_size=train_loader.batch_size,
               pretrain_epoch=PRE_EPOCHS)


torch.save(model.state_dict(), "./model_cov_pretrain.pt")

# 2) self-training + clustering fine-tuning
pred_labels = model.finetrain(train_loader,
                              batch_size=train_loader.batch_size,
                              train_epoch=FINET_EPOCHS,
                              update_epoch=UPDATE_EVERY)


# Specify the path to save the model
PATH = "./model_cov.pt"

# Save the model's state_dict
torch.save(model.state_dict(), PATH)

