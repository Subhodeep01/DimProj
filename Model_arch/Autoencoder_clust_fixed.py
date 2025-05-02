import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from LID_estimator import estimate_id_torch_soft
from cluster_kl import cal_dist, cal_latent, target_dis
import numpy as np
from sklearn.cluster import KMeans

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    """Auto‑encoder + DEC/IDEC clustering.

    Args
    -----
    cluster_num : int
        Number of centroids (K‑means clusters).
    t_alpha, alpha, gamma : float
        Hyper‑parameters exactly as in the original TensorFlow code.
    lr_rate : float
        Learning‑rate for Adam.
    input_dim, hidden_dim, bottleneck_dim : int
        Network width parameters.
    """

    def __init__(self, *, cluster_num: int, t_alpha: float, alpha: float, gamma: float,
                 lr_rate: float, input_dim: int = 784, hidden_dim1: int = 256, hidden_dim2: int = 128,
                 bottleneck_dim: int = 32):
        super().__init__()
        self.t_alpha = t_alpha
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.gamma = gamma
        self.bottleneck_dim = bottleneck_dim
        self.lr_rate = lr_rate

        # learnable centroids ------------------------------------------------
        self.clusters = nn.Parameter(torch.empty(cluster_num, bottleneck_dim))
        init.xavier_uniform_(self.clusters)

        # dense AE weights (He init) ----------------------------------------
        self.W1 = nn.Parameter(torch.randn(hidden_dim1, input_dim) * (2 / input_dim) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim1))
        
        self.W2 = nn.Parameter(torch.randn(hidden_dim2, hidden_dim1) * (2 / hidden_dim1) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.W3 = nn.Parameter(torch.randn(bottleneck_dim, hidden_dim2) * (2 / hidden_dim2) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(bottleneck_dim))

        self.W4 = nn.Parameter(torch.randn(hidden_dim2, bottleneck_dim) * (2 / bottleneck_dim) ** 0.5)
        self.b4 = nn.Parameter(torch.zeros(hidden_dim2))
        
        self.W5 = nn.Parameter(torch.randn(hidden_dim1, hidden_dim2) * (2 / hidden_dim2) ** 0.5)
        self.b5 = nn.Parameter(torch.zeros(hidden_dim1))
        
        self.W6 = nn.Parameter(torch.randn(input_dim, hidden_dim1) * (2 / hidden_dim1) ** 0.5)
        self.b6 = nn.Parameter(torch.zeros(input_dim))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, self_training: bool = False):
        # encoder -----------------------------------------------------------
        a1 = torch.relu(x @ self.W1.t() + self.b1)
        a2 = torch.relu(a1 @ self.W2.t() + self.b2)
        encoded = torch.relu(a2 @ self.W3.t() + self.b3)

        # latent‑space objectives ------------------------------------------
        num, latent_p = cal_latent(encoded, self.t_alpha)
        latent_q = target_dis(latent_p)
        diag = torch.diagonal(num)
        latent_p = latent_p + torch.diag_embed(diag)
        latent_q = latent_q + torch.diag_embed(diag)

        latent_dist1, latent_dist2 = cal_dist(encoded, self.clusters)
        kmeans_loss = latent_dist2.sum(dim=1).mean()

        # decoder -----------------------------------------------------------
        a3 = torch.relu(encoded @ self.W4.t() + self.b4)
        a4 = torch.relu(a3 @ self.W5.t() + self.b5)
        recon = torch.sigmoid(a4 @ self.W6.t() + self.b6)
        recon_loss = ((recon - x) ** 2).mean()

        if self_training:
            eps = 1e-10
            rho = estimate_id_torch_soft(latent_p) / estimate_id_torch_soft(latent_q)
            kl_loss = F.kl_div(latent_p.log(), latent_q, reduction="batchmean")
            ISloss = rho - torch.log(rho + eps) - 1.0
            total = recon_loss + self.alpha * kmeans_loss + self.gamma * kl_loss + 0.001*ISloss
        else:
            kl_loss = torch.tensor(0.0, device=x.device)
            total = recon_loss

        return total, encoded, latent_dist1

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _init_latent_buffer(self, N: int, device):
        self.latent_repre = torch.zeros(N, self.bottleneck_dim, device=device)

    @torch.no_grad()
    def _push_centroids_to_param(self, centers: np.ndarray):
        self.clusters.data.copy_(torch.as_tensor(centers, dtype=self.clusters.dtype, device=self.clusters.device))

    # ------------------------------------------------------------------
    # stage‑1: reconstruction pre‑training
    # ------------------------------------------------------------------
    def pretrain(self, train_loader: DataLoader, batch_size: int, pretrain_epoch: int):
        """Pre‑train the AE with only reconstruction loss.

        `batch_size` is accepted to keep the old call signature. The DataLoader
        already contains its own `batch_size`; if the two differ we issue a
        warning but proceed with the loader’s setting.
        """
        if train_loader.batch_size != batch_size:
            print(f"[warn] DataLoader batch_size={train_loader.batch_size} but argument batch_size={batch_size}; proceeding with loader setting.")

        self.to(DEVICE)
        opt = optim.Adam(self.parameters(), lr=self.lr_rate)
        self._init_latent_buffer(len(train_loader.dataset), DEVICE)
        self.train()

        for epoch in range(pretrain_epoch):
            for idx, x, _ in train_loader:
                idx = idx.to(DEVICE, non_blocking=True)
                x = x.to(DEVICE, non_blocking=True)
                loss, latent, _ = self.forward(x, self_training=False)
                opt.zero_grad(); loss.backward(); opt.step()
                self.latent_repre[idx] = latent.detach()

            print(f"[pre‑train] epoch {epoch+1}/{pretrain_epoch}  recon={loss.item():.4f}")

        self.latent_repre = self.latent_repre.cpu().numpy()

    # ------------------------------------------------------------------
    # stage‑2: joint fine‑tuning with clustering
    # ------------------------------------------------------------------
    def finetrain(self, train_loader: DataLoader, batch_size: int, train_epoch: int,
                  update_epoch: int, *, error: float = 1e-3):
        """Fine‑tune with clustering losses.
        """
        if train_loader.batch_size != batch_size:
            print(f"[warn] DataLoader batch_size={train_loader.batch_size} but argument batch_size={batch_size}; proceeding with loader setting.")

        self.to(DEVICE)
        opt = optim.Adam(self.parameters(), lr=self.lr_rate)

        # --- (1) initial K‑means -----------------------------------------
        kmeans = KMeans(n_clusters=self.cluster_num, init="k-means++", n_init="auto").fit(np.nan_to_num(self.latent_repre))
        self.last_pred = kmeans.labels_.copy()
        self._push_centroids_to_param(kmeans.cluster_centers_)

        # --- (2) training loop ------------------------------------------
        for epoch in range(train_epoch):
            if epoch % update_epoch == 0:
                self.eval()
                N = len(train_loader.dataset)
                Y_pred = np.empty(N, dtype=np.int32)

                with torch.no_grad():
                    for idx, x, _ in train_loader:
                        idx = idx.to(DEVICE, non_blocking=True)
                        x = x.to(DEVICE, non_blocking=True)
                        _, _, dist_b = self.forward(x, self_training=True)
                        Y_pred[idx.cpu().numpy()] = dist_b.argmin(dim=1).cpu().numpy()

                change_rate = np.mean(Y_pred != self.last_pred)
                print(f"[fine‑train] epoch {epoch:03d}  change‑rate = {change_rate:.4f}")

                if change_rate < error:
                    print("early‑stop: label change below threshold")
                    break
                self.last_pred = Y_pred

            # mini‑batch optimisation -----------------------------------
            self.train()
            for _, x, _ in train_loader:
                x = x.to(DEVICE, non_blocking=True)
                opt.zero_grad()
                total_loss, _, _ = self.forward(x, self_training=True)
                total_loss.backward(); opt.step()

            print(f"[fine‑train] epoch {epoch+1}/{train_epoch}  total={total_loss.item():.4f}")

        return self.last_pred
