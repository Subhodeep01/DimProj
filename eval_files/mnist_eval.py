import torch
import numpy as np
from Autoencoder_clust import Autoencoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datasetprep import train_loader               # code above      
from sklearn.cluster import KMeans       
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- prerequisites -----------------------------------------------------------
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
        input_dim   = input_dim,           # ? **critical change**
        hidden_dim  = 64,
        bottleneck_dim = 3).to(device)

model.load_state_dict(torch.load("model_mnist.pt"))  # Load weights

# ---------------- 1.  collect ALL latent vectors ----------------------------
model.eval()                                   # inference mode
device = next(model.parameters()).device

all_latent = []                                # list of tensors (B, bottleneck_dim)
all_labels = []                                # corresponding cluster label
all_true_labels = []

with torch.no_grad():
    for idx, x, true_labels in train_loader:             # same loader you trained with
        x = x.to(device, non_blocking=True)

        _, latent_batch, _ = model.forward(x, self_training=False)
        all_latent.append(latent_batch.cpu())
        
        # fetch predicted cluster for this batch
        # (re-run the distance calc on GPU for the current batch)
        dist_batch = model.latent_dist1.cpu()  # created inside forward()
        all_labels.append(dist_batch.argmin(dim=1))
        all_true_labels.append(true_labels.cpu())

# (N, bottleneck_dim)  &  (N,)
latent_full  = torch.cat(all_latent,  dim=0).numpy()
labels_full  = torch.cat(all_labels, dim=0).numpy()
K            = np.unique(labels_full).size
true_labels_full = torch.cat(all_true_labels,  dim=0).numpy()



# 2)  run K-means ------------------------------------------------------------
K          = 10                         # or set your own, e.g. K = 8
km         = KMeans(n_clusters=K, init="k-means++", n_init="auto", random_state=0)
labels_km  = km.fit_predict(latent_full)
# ---------------- 2.  t-SNE --------------------------------------------------

#tsne = TSNE(n_components=3, perplexity=30, learning_rate="auto",
 #           init="pca", max_iter=1000, random_state=0)
#xyz = tsne.fit_transform(latent_full)           # (N, 2)


# ----------------------------------------------------------------
# 3)  3-D scatter plot
# ----------------------------------------------------------------
fig = plt.figure(figsize=(9, 7), dpi=120)
ax  = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                     c=true_labels_full, s=20, alpha=0.85)

ax.set_title(f"MNIST 3-D finetrained autoencoder latent space with true labels K = {K}", pad=18)
ax.set_xlabel("Latent-1")
ax.set_ylabel("Latent-2")
ax.set_zlabel("Latent-3")
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

# optional legend (good for = 12 clusters)
leg_elems = [Line2D([0], [0], marker='o', linestyle='',
                    label=f"Cluster {k}",
                    markerfacecolor=scatter.cmap(scatter.norm(k)))
             for k in range(K)]
ax.legend(handles=leg_elems, title="k-means label", loc="upper left")

fig.tight_layout()



print(f"3-D figure saved  ")
plt.savefig("mnist_truelblsfinetrained3d10c.png", dpi=300, bbox_inches='tight')

plt.show()