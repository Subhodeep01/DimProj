import torch
import numpy as np
from Autoencoder_clust import Autoencoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from diabetes_prep import train_loader, features               # code above      
from sklearn.cluster import KMeans       
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- prerequisites -----------------------------------------------------------
model = Autoencoder(
        cluster_num = 4,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 5*1e-3,
        input_dim   = features.shape[1],           # ? **critical change**
        hidden_dim  = 64,
        bottleneck_dim = 3
).to(device)

model.load_state_dict(torch.load("model_diabetes_pretrain.pt"))  # Load weights

# ---------------- 1.  collect ALL latent vectors ----------------------------
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



# 2)  run K-means ------------------------------------------------------------
K          = 4                         # or set your own, e.g. K = 8
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
                     c=labels_km, s=20, alpha=0.85)

ax.set_title(f"Diabetes 3-D just autoencoder latent space K = {K}", pad=18)
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
plt.savefig("diabetes_pretrained3d4c.png", dpi=300, bbox_inches='tight')

plt.show()