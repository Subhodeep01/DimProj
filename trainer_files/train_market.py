# --- prerequisites -----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Autoencoder_clust_fixed import Autoencoder               
from preprocess_market import build_loader               
from sklearn.manifold import TSNE
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------ data ---------------
train_loader, input_dim, scaler = build_loader("./data/Marketing/marketing_campaign.csv",
                                               batch_size=64)

# ------------ model --------------
model = Autoencoder(cluster_num=4,        # K
                    t_alpha=1.0,
                    alpha=0.001,
                    gamma=0.001,
                    lr_rate=1e-3,
                    input_dim=input_dim,   # <- **36** with defaults above
                    hidden_dim=64,
                    bottleneck_dim=3).to(device)

# ------------ training -----------
model.pretrain(train_loader,
               batch_size=64,
               pretrain_epoch=50)

torch.save(model.state_dict(), "./model_market_pretrain.pt")

pred_labels = model.finetrain(train_loader,
                              batch_size=64,
                              train_epoch=200,
                              update_epoch=5)


# Specify the path to save the model
PATH = "./model_market.pt"

# Save the model's state_dict
torch.save(model.state_dict(), PATH)

