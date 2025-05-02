import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from LID_estimator import estimate_id_torch_soft
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def local_two_nn_id(x1d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x1d : (N, 1) column vector, requires_grad OK
    returns scalar tensor = mean 1/log(r2 / r1)
    """
    dist = torch.cdist(x1d, x1d) + eps               # (N,N)
    d_sorted, _ = torch.topk(dist, k=3, largest=False)
    r1, r2 = d_sorted[:, 1], d_sorted[:, 2]
    lid = 1.0 / (torch.log((r2 + eps) / (r1 + eps)) + eps)
    return lid.mean()                                # scalar tensor


def featurewise_id_sum_local(X: torch.Tensor) -> torch.Tensor:
    ids = torch.stack([local_two_nn_id(X[:, i:i+1]) for i in range(X.shape[1])])
    return ids, ids.sum()
# -------------------------------------------------
# Autoencoder definition
# -------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, bottleneck_dim=32, _lambda=0.001):
        super().__init__()
        self._lambda = _lambda
        self.bdim = bottleneck_dim
        # He initialisation for ReLU layers
        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * (2 / input_dim) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(bottleneck_dim, hidden_dim) * (2 / hidden_dim) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(bottleneck_dim))

        self.W3 = nn.Parameter(torch.randn(hidden_dim, bottleneck_dim) * (2 / bottleneck_dim) ** 0.5)
        self.b3 = nn.Parameter(torch.zeros(hidden_dim))
        self.W4 = nn.Parameter(torch.randn(input_dim, hidden_dim) * (2 / hidden_dim) ** 0.5)
        self.b4 = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # Encoder
        z1 = x @ self.W1.t() + self.b1
        a1 = torch.relu(z1)
        z2 = a1 @ self.W2.t() + self.b2
        self.encoded = torch.relu(z2)

        # Decoder
        z3 = self.encoded @ self.W3.t() + self.b3
        a3 = torch.relu(z3)
        z4 = a3 @ self.W4.t() + self.b4
        self.decoded = torch.sigmoid(z4)
        return self.decoded, self.encoded

    def recon_loss(self, recon, target):
        lid = estimate_id_torch_soft(self.encoded)
       # ids, total = featurewise_id_sum_local(self.encoded)
        lidloss = self._lambda*(self.bdim - lid)
        recon_loss = ((recon - target) ** 2).mean()
        return recon_loss + lidloss

