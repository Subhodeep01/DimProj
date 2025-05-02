import torch
import numpy as np
from Autoencoder_sparse_clust import Autoencoder
from sparse_dataset_prep import train_loader, X_sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Hyper‑parameters & data
# -------------------------------------------------
input_dim = X_sparse.size(1)
hidden_dim = 128
bottleneck_dim = 32
batch_size = 64
epochs = 10
learning_rate = 1e-3
cluster_num = 10 
t_alpha = 1.0
alpha = 0.001
gamma = 0.001
self_training = False
pretrain_epoch = 10
train_epoch = 100
update_epoch = 5

lid_tau = 1.0    # soft sort temperature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = Autoencoder(cluster_num, t_alpha, alpha, gamma, learning_rate, input_dim, hidden_dim, bottleneck_dim).to(device)
model1.pretrain(train_loader, batch_size, 10)
# Specify the path to save the model
PATH = "./model1_sparse.pt"

# Save the model's state_dict
torch.save(model1.state_dict(), PATH)


model2 = Autoencoder(cluster_num, t_alpha, alpha, gamma, learning_rate, input_dim, hidden_dim, bottleneck_dim).to(device)
model2.pretrain(train_loader, batch_size, pretrain_epoch)
y_pred = model2.train(train_loader, batch_size, train_epoch, update_epoch,)
PATH = "./model2_sparse.pt"

# Save the model's state_dict
torch.save(model2.state_dict(), PATH)
# Y = [y for idx, x, y in train_loader]
# Y = np.array(Y)
# ARI = np.around(adjusted_rand_score(Y, y_pred), 5)
# NMI = np.around(normalized_mutual_info_score(Y, y_pred), 5)

# print("ARI: ", ARI, "    NMI: ", NMI)
# Y = [y for idx, x, y in train_loader]
# Y = np.array(Y)
# ARI = np.around(adjusted_rand_score(Y, y_pred), 5)
# NMI = np.around(normalized_mutual_info_score(Y, y_pred), 5)
