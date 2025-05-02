import torch
import numpy as np
from Autoencoder_clust import Autoencoder
from datasetprep import train_loader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Hyper-parameters & data
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
pretrain_epoch = 10
train_epoch = 25
update_epoch = 30

lid_tau = 1.0    # soft sort temperature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder(cluster_num, t_alpha, alpha, gamma, learning_rate, input_dim, hidden_dim, bottleneck_dim).to(device)
model.load_state_dict(torch.load("model2.pt"))  # Load weights
#model.eval()  #evaluate the model

#all_encodings = []
#all_labels = []
#all_recon = []

#with torch.no_grad():
 #   for idx, images, labels in train_loader:
  #      images = images.to(device)
   #     z1 = images @ model.W1.t() + model.b1
    #    a1 = torch.relu(z1)
     #   z2 = a1 @ model.W2.t() + model.b2
      #  encoded = torch.relu(z2)
       # all_encodings.append(encoded.cpu())
        #all_labels.append(labels)
        
        

#all_encodings = torch.cat(all_encodings, dim=0)
#all_labels = torch.cat(all_labels, dim=0)
model.eval()                                   # inference mode
device = next(model.parameters()).device

all_latent = []                                # list of tensors (B, bottleneck_dim)
all_labels = []                                # corresponding cluster label

with torch.no_grad():
    for idx, x, _ in train_loader:             # same loader you trained with
        x = x.to(device, non_blocking=True)

        _, latent_batch, _ = model.forward(x, self_training=False)
        all_latent.append(latent_batch.cpu())
        
        # fetch predicted cluster for this batch
        # (re-run the distance calc on GPU for the current batch)
        dist_batch = model.latent_dist1.cpu()  # created inside forward()
        all_labels.append(dist_batch.argmin(dim=1))

# (N, bottleneck_dim)  &  (N,)
latent_full  = torch.cat(all_latent,  dim=0).numpy()
labels_full  = torch.cat(all_labels, dim=0).numpy()
K            = np.unique(labels_full).size

#tsne = TSNE(n_components=2)
#xyz = tsne.fit_transform(latent_full)

#tsne2 = TSNE(n_components = 3)
#recon_2d = tsne.fit_transform(all_recon)

#plt.figure(figsize=(8, 6))
#plt.scatter(xyz[:, 0], xyz[:, 1], c=labels_full, cmap='tab10', alpha=0.7)
#plt.colorbar()
#plt.title("t-SNE of Latent Space")

#plt.savefig("tsne_plot3.png", dpi=300, bbox_inches='tight')


#plt.show()
fig = plt.figure(figsize=(9, 7), dpi=120)
ax  = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                     c=labels_full, s=20, alpha=0.85)

ax.set_title(f"3-D latent space K = {K}", pad=18)
ax.set_xlabel("latent-1")
ax.set_ylabel("latent-2")
ax.set_zlabel("latent-3")
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

# optional legend (good for = 12 clusters)
leg_elems = [Line2D([0], [0], marker='o', linestyle='',
                   label=f"Cluster {k}",
                    markerfacecolor=scatter.cmap(scatter.norm(k)))
             for k in range(K)]
ax.legend(handles=leg_elems, title="k-means label", loc="upper left")

fig.tight_layout()



print(f"3-D figure saved  ")
plt.savefig("mnist_3d.png", dpi=300, bbox_inches='tight')

plt.show()