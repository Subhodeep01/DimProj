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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Autoencoder definition
# -------------------------------------------------
class SparseLinear(nn.Module):
    """Linear layer that accepts input in sparse COO format."""
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
    def forward(self, x):                                  # x sparse
        out = torch.sparse.mm(x, self.weight.t())          # (B, out_dim)
        if self.bias is not None:
            out = out + self.bias
        return out
    
class Autoencoder(nn.Module):
    def __init__(self, cluster_num, t_alpha, alpha, gamma, lr_rate, input_dim=784, hidden_dim=128, bottleneck_dim=32, ):
        super().__init__()
        self.t_alpha = t_alpha
        self.clusters = nn.Parameter(
            torch.empty(cluster_num, bottleneck_dim),  # (K, latent_dim)
            requires_grad=True
        )
        init.xavier_uniform_(self.clusters)      # Glorot-uniform init
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.gamma = gamma
        self.bottleneck_dim = bottleneck_dim
        self.lr_rate = lr_rate

        self.enc1 = SparseLinear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, bottleneck_dim)
        self.dec1 = nn.Linear(bottleneck_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)      # dense decoder
        # self.W1 = nn.Parameter(torch.randn(hidden_dim2, input_dim) * (2 / input_dim) ** 0.5)
        # self.b1 = nn.Parameter(torch.zeros(hidden_dim2))
        # self.W2 = nn.Parameter(torch.randn(bottleneck_dim, hidden_dim) * (2 / hidden_dim) ** 0.5)
        # self.b2 = nn.Parameter(torch.zeros(bottleneck_dim))

        # self.W3 = nn.Parameter(torch.randn(hidden_dim, bottleneck_dim) * (2 / bottleneck_dim) ** 0.5)
        # self.b3 = nn.Parameter(torch.zeros(hidden_dim))
        # self.W4 = nn.Parameter(torch.randn(input_dim, hidden_dim) * (2 / hidden_dim) ** 0.5)
        # self.b4 = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, self_training = False):
        self.self_training = self_training

        # Encoder
        h1 = F.relu(self.enc1(x))
        encoded  = F.relu(self.enc2(h1))

        # Latent space
        self.latent = encoded
        self.num, self.latent_p = cal_latent(self.latent, self.t_alpha)
        self.latent_q = target_dis(self.latent_p)

        diag = torch.diagonal(self.num)
        self.latent_p = self.latent_p + torch.diag_embed(diag)
        self.latent_q = self.latent_q + torch.diag_embed(diag)

        self.latent_dist1, self.latent_dist2 = cal_dist(self.latent, self.clusters)
        self.kmeans_loss = self.latent_dist2.sum(dim=1).mean()          # ⭠ tf.reduce_sum ▸ tf.reduce_mean

        # Decoder
        # decoder: **no sigmoid**
        h2 = F.relu(self.dec1(encoded))
        recon_logits = self.dec2(h2)                 # raw scores, any range
        loss = nn.BCEWithLogitsLoss(reduction="sum")(recon_logits,
                                                     x.to_dense()) / x.size(0)

        self.recon_loss = loss

        if self.self_training:
            eps = 1e-10                                            # avoid log(0)
            # cross_entropy = -(self.latent_q * (self.latent_p + eps).log()).sum()
            # entropy        = -(self.latent_q * (self.latent_q + eps).log()).sum()
            self.rho = estimate_id_torch_soft(self.latent_p) / estimate_id_torch_soft(self.latent_q)
            self.kl_loss        = self.rho - torch.log(self.rho+eps) - 1    # Itakura-Saito divergence

            total_loss = (self.recon_loss +
                          self.alpha * self.kmeans_loss +
                          self.gamma * self.kl_loss)
        else:
            total_loss = self.recon_loss

        return total_loss, self.latent, self.latent_dist1
    
    @torch.no_grad()
    def _init_latent_buffer(self, N, device):
        """Allocate a tensor to collect all latent vectors."""
        self.latent_repre = torch.zeros(N, self.bottleneck_dim, device=device)

    @torch.no_grad()
    def _push_centroids_to_param(self, param: torch.nn.Parameter, centers: np.ndarray):
        """Copy scikit-learn K-means centroids into the model parameter."""
        src = torch.as_tensor(centers, dtype=param.dtype, device=param.device)
        param.data.copy_(src)


    def pretrain(self, train_loader, batch_size: int, pretrain_epoch: int):

        print("begin the pretraining")

        # ---------------  device / GPU selection  ----------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_rate)
        num_samples = len(train_loader.dataset)

        # ---------------  training loop  -------------------------------------
        self.train()                                           # enable grads
        self._init_latent_buffer(num_samples, device)

        for epoch in range(pretrain_epoch):
            epoch_loss = 0.0
            for idx, x, _ in train_loader:
                idx = idx.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                #print(x[:5])
                recon_loss, latent, _ = self.forward(x)
                # zero-grad → forward → backward → Adam step
                self.optimizer.zero_grad()
                recon_loss.backward()
                # after loss.backward()
                #grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float('inf'))
                #print(f"loss={recon_loss.item():.4f}, grad_norm={grad_norm:.4f}")
                self.optimizer.step()

                # stash latent vectors in the master buffer
                self.latent_repre[idx] = latent.detach()

            print(f"[epoch {epoch+1:02d}/{pretrain_epoch}] "
                f"recon_loss = {recon_loss.item():.4f}")

        # bring latent buffer back to CPU for later K-means
        self.latent_repre = self.latent_repre.cpu().numpy()

    def finetrain(self, train_loader, batch_size: int, train_epoch: int, update_epoch: int, error:float=0.001):
        self_training = True
        print("begin the training")

        # ---------------  device / GPU selection  ----------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_rate)

            # ---------------- 1) initial K-means in the pretrained space --------
        latent_clean = np.nan_to_num(self.latent_repre)          # from pre-train step
        kmeans       = KMeans(n_clusters=self.cluster_num,
                            init="k-means++",
                            n_init="auto").fit(latent_clean)

        self.kmeans_pred = kmeans.labels_
        self.last_pred   = self.kmeans_pred.copy()
        self._push_centroids_to_param(self.clusters, kmeans.cluster_centers_)

        # ---------------  training loop  -------------------------------------
        for epoch in range(train_epoch):
            if epoch % update_epoch == 0:
                self.eval()
                all_dist = []
                with torch.no_grad():
                    for idx, x, _ in train_loader:
                        x = x.to(device, non_blocking=True)
                        recon_loss, latent, dist = self.forward(x, self_training)
                        all_dist.append(dist.cpu())               

                dist = torch.cat(all_dist, dim=0).numpy()        # (N, K)
                self.Y_pred = dist.argmin(axis=1)

                change_rate = np.mean(self.Y_pred != self.last_pred)
                print(f"[epoch {epoch:03d}] change-rate = {change_rate:.4f}")

                if change_rate < error:
                    print("early-stop: label change below threshold")
                    break
                else:
                    self.last_pred = self.Y_pred

            # ---- 2.B  one epoch of mini-batch optimisation -----------------
            self.train()
            for idx, x, _ in train_loader:
                x = x.to(device, non_blocking=True)

                # zero-grad → forward → backward → Adam step
                self.optimizer.zero_grad()
                total_loss, latent, _ = self.forward(x, self_training)
                total_loss.backward()
                self.optimizer.step()
                
            print(f"[epoch {epoch+1:02d}/{train_epoch}] "
                f"total_loss = {total_loss.item():.4f}")

        return self.Y_pred
    
    # def test(self, test_loader):
    #     all_dist = []
    #     with torch.no_grad():
    #         for idx, x, _ in train_loader:
    #             x = x.to(device)
    #             recon_loss, latent, dist = self.forward(x, self_training)
    #             all_dist.append(dist.cpu())               

    #         dist = torch.cat(all_dist, dim=0).numpy()        # (N, K)
    #         self.Y_pred = dist.argmin(axis=1)

    #         change_rate = np.mean(self.Y_pred != self.last_pred)
    #         print(f"[epoch {epoch:03d}] change-rate = {change_rate:.4f}")

    #         if change_rate < error:
    #             print("early-stop: label change below threshold")
    #         else:
    #             self.last_pred = self.Y_pred