import torch
import numpy as np
from Autoencoder_clust_fixed import Autoencoder
from datasetprep import train_loader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Hyper‑parameters & data
# -------------------------------------------------
input_dim = 28 * 28
hidden_dim = 64
bottleneck_dim = 3
batch_size = 64
epochs = 10
learning_rate = 1e-3
cluster_num = 10 
t_alpha = 1.0
alpha = 0.001
gamma = 0.001
self_training = False
pretrain_epoch = 50
train_epoch = 200
update_epoch = 5

lid_tau = 1.0    # soft sort temperature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder(cluster_num = cluster_num,            # or whatever you need
        t_alpha     = t_alpha,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 1e-3,
        input_dim   = input_dim,           # ← **critical change**
        hidden_dim  = 64,
        bottleneck_dim = 3).to(device)
model.pretrain(train_loader, batch_size, pretrain_epoch)


#num_samples = len(train_loader.dataset)
#labels_buf  = torch.empty(num_samples, dtype=torch.long)

#with torch.no_grad():
#    for idx_b, _, lbl_b in train_loader:          # (index, image, label)
#        labels_buf[idx_b] = lbl_b                 ## place each label at its slot

# Specify the path to save the model
PATH = "./model_mnist_pretrain.pt"

# Save the model's state_dict
torch.save(model.state_dict(), PATH)

y_pred = model.finetrain(train_loader, batch_size, train_epoch, update_epoch,)
# Specify the path to save the model
PATH = "./model_mnist.pt"

# Save the model's state_dict
torch.save(model.state_dict(), PATH)
# Y = [y for idx, x, y in train_loader]
# Y = np.array(Y)
# ARI = np.around(adjusted_rand_score(Y, y_pred), 5)
# NMI = np.around(normalized_mutual_info_score(Y, y_pred), 5)

