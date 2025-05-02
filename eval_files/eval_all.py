import torch
import numpy as np
from Autoencoder_clust import Autoencoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ship_prep import train_loader, features               # code above      
from sklearn.cluster import KMeans       
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ship evaluation -----------------------------------------------------------
model7 = Autoencoder(
        cluster_num = 4,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 5*1e-3,
        input_dim   = features.shape[1],           # ? **critical change**
        hidden_dim  = 64,
        bottleneck_dim = 3
).to(device)

model8 = Autoencoder(
        cluster_num = 4,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 5*1e-3,
        input_dim   = features.shape[1],           # ? **critical change**
        hidden_dim  = 64,
        bottleneck_dim = 3
).to(device)

model7.load_state_dict(torch.load("model_ship_pretrain.pt"))  # Load weights

model8.load_state_dict(torch.load("model_ship.pt"))  # Load weights

models = [model7, model8]

which = ["pretrained", "trained"]

# ---------------- 1.  collect ALL latent vectors ----------------------------
i = 0
for model in models:
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
  # ----------------------------------------------------------------
  # 3)  3-D scatter plot
  # ----------------------------------------------------------------
  fig = plt.figure(figsize=(9, 7), dpi=120)
  ax  = fig.add_subplot(111, projection='3d')
  if i == 0:
    scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                         c=labels_km, s=20, alpha=0.85)
  elif i == 1:
    scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                         c=labels_full, s=20, alpha=0.85)
  
  ax.set_title(f"Ship 3-D {which[i]} autoencoder latent space K = {K}", pad=18)
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
  plt.savefig(f"ship_{which[i]}3d.png", dpi=300, bbox_inches='tight')
  i+=1



# ---------------- Cover type evaluation ----------------------------
from covtype_prep import train_loader

model1 = Autoencoder(
        cluster_num = 7,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 1e-4,
        input_dim   = 54,           # ? **critical change**
        hidden_dim  = 32,
        bottleneck_dim = 3
).to(device)
model1.load_state_dict(torch.load("model_cov_pretrain.pt"))  # Load weights

model2 = Autoencoder(
        cluster_num = 7,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 1e-4,
        input_dim   = 54,           # ? **critical change**
        hidden_dim  = 32,
        bottleneck_dim = 3
).to(device)
model2.load_state_dict(torch.load("model_cov.pt"))  # Load weights

models = [model1, model2]

which = ["pretrained", "trained"]

# ---------------- 1.  collect ALL latent vectors ----------------------------
i = 0
for model in models:
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
  K          = 7                         # or set your own, e.g. K = 8
  km         = KMeans(n_clusters=K, init="k-means++", n_init="auto", random_state=0)
  labels_km  = km.fit_predict(latent_full)
  # ----------------------------------------------------------------
  # 3)  3-D scatter plot
  # ----------------------------------------------------------------
  fig = plt.figure(figsize=(9, 7), dpi=120)
  ax  = fig.add_subplot(111, projection='3d')
  if i == 0:
    scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                         c=labels_km, s=20, alpha=0.85)
  elif i == 1:
    scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                         c=labels_full, s=20, alpha=0.85)
  
  ax.set_title(f"Cover type 3-D {which[i]} autoencoder latent space K = {K}", pad=18)
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
  plt.savefig(f"Cover_{which[i]}3d.png", dpi=300, bbox_inches='tight')
  i+=1




# ---------------- Market evaluation ----------------------------
from preprocess_market import build_loader
train_loader, input_dim, scaler = build_loader("./data/Marketing/marketing_campaign.csv",
                                               batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model3 = Autoencoder(cluster_num=4,        # K
                    t_alpha=1.0,
                    alpha=0.001,
                    gamma=0.001,
                    lr_rate=1e-3,
                    input_dim=input_dim,  
                    hidden_dim=64,
                    bottleneck_dim=3).to(device)
model3.load_state_dict(torch.load("model_market_pretrain.pt"))  # Load weights

model4 = Autoencoder(cluster_num=4,        # K
                    t_alpha=1.0,
                    alpha=0.001,
                    gamma=0.001,
                    lr_rate=1e-3,
                    input_dim=input_dim,  
                    hidden_dim=64,
                    bottleneck_dim=3).to(device)
model4.load_state_dict(torch.load("model_market.pt"))  # Load weights

models = [model3, model4]

which = ["pretrained", "trained"]

i = 0
for model in models:
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
  # ----------------------------------------------------------------
  # 3)  3-D scatter plot
  # ----------------------------------------------------------------
  fig = plt.figure(figsize=(9, 7), dpi=120)
  ax  = fig.add_subplot(111, projection='3d')
  if i == 0:
    scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                         c=labels_km, s=20, alpha=0.85)
  elif i == 1:
    scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                         c=labels_full, s=20, alpha=0.85)
  
  ax.set_title(f"Market 3-D {which[i]} autoencoder latent space K = {K}", pad=18)
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
  plt.savefig(f"Market_{which[i]}3d.png", dpi=300, bbox_inches='tight')
  i+=1
  
  
  
  
  
# ----------------Diabetes evaluation ----------------------------
from diabetes_prep import train_loader, features
model5 = Autoencoder(
        cluster_num = 4,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 5*1e-3,
        input_dim   = features.shape[1],           # ? **critical change**
        hidden_dim  = 64,
        bottleneck_dim = 3
).to(device)

model5.load_state_dict(torch.load("model_diabetes_pretrain.pt"))  # Load weights

model6 = Autoencoder(
        cluster_num = 4,            # or whatever you need
        t_alpha     = 1.0,
        alpha       = 0.001,
        gamma       = 0.001,
        lr_rate     = 5*1e-3,
        input_dim   = features.shape[1],           # ? **critical change**
        hidden_dim  = 64,
        bottleneck_dim = 3
).to(device)

model6.load_state_dict(torch.load("model_diabetes.pt"))  # Load weights

models = [model5, model6]

which = ["pretrained", "trained"]

i = 0
for model in models:
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
  # ----------------------------------------------------------------
  # 3)  3-D scatter plot
  # ----------------------------------------------------------------
  fig = plt.figure(figsize=(9, 7), dpi=120)
  ax  = fig.add_subplot(111, projection='3d')
  if i == 0:
    scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                         c=labels_km, s=20, alpha=0.85)
  elif i == 1:
    scatter = ax.scatter(latent_full[:, 0], latent_full[:, 1], latent_full[:, 2],
                         c=labels_full, s=20, alpha=0.85)
  
  ax.set_title(f"Diabetes 3-D {which[i]} autoencoder latent space K = {K}", pad=18)
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
  plt.savefig(f"Diabetes_{which[i]}3d.png", dpi=300, bbox_inches='tight')
  i+=1
  

