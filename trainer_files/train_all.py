# --- prerequisites -----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Autoencoder_clust_fixed import Autoencoder               # same folder
from diabetes_prep import train_loader, features             # code above
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder(
        cluster_num = 4,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 1e-3,
        input_dim   = features.shape[1],           # ? **critical change**
        hidden_dim  = 64,
        bottleneck_dim = 3
).to(device)

PRE_EPOCHS   = 100
FINET_EPOCHS = 200
UPDATE_EVERY = 10

# 1) unsupervised reconstruction pre-training
model.pretrain(train_loader,
               batch_size=train_loader.batch_size,
               pretrain_epoch=PRE_EPOCHS)
               
torch.save(model.state_dict(), "./model_diabetes_pretrain.pt")

# 2) self-training + clustering fine-tuning
pred_labels = model.finetrain(train_loader,
                              batch_size=train_loader.batch_size,
                              train_epoch=FINET_EPOCHS,
                              update_epoch=UPDATE_EVERY)

# Specify the path to save the model
PATH = "./model_diabetes.pt"



# Save the model's state_dict
torch.save(model.state_dict(), PATH)


from ship_prep import train_loader, features

model = Autoencoder(
        cluster_num = 4,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 1e-3,
        input_dim   = features.shape[1],           # ? **critical change**
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



from preprocess_market import build_loader

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



from covtype_prep import train_loader


model = Autoencoder(
        cluster_num = 7,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 1e-4,
        input_dim   = 54,           # ? **critical change**
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

