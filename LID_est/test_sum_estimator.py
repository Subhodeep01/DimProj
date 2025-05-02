"""
check_featurewise_id.py
-----------------------
Demo: per-feature Two-NN IDs with estimate_id_torch_soft
"""

import torch
from LID_estimator import estimate_id_torch_soft           # your function
import torchsort                                           # ensure it's installed

def featurewise_id_sum(X, tau=0.5):
    """
    Compute ID for every column separately and return their tensor + sum.
    """
    ids = [estimate_id_torch_soft(X[:, i:i+1], tau=tau)    # (N,1) slice
           for i in range(X.shape[1])]
    ids = torch.stack(ids)                                 # (D,)
    return ids, ids.sum()                                  # (D,), scalar-tensor

# ----------------------------------------------------------------------
# 1. make some toy data:  3-D latent ? 8-D observed  + small noise
# ----------------------------------------------------------------------
N, latent_d, D = 512, 3, 8
device = "cuda" if torch.cuda.is_available() else "cpu"

latent  = torch.randn(N, latent_d, device=device)
proj    = torch.randn(latent_d, D, device=device)
X       = (latent @ proj + 0.05 * torch.randn(N, D, device=device)
          ).requires_grad_(True)                           # wants grads

# ----------------------------------------------------------------------
# 2. feature-wise IDs  +  sum  + back-prop check
# ----------------------------------------------------------------------
ids, total_id = featurewise_id_sum(X, tau=0.5)
print("Per-feature IDs:", ids.detach().cpu().numpy())
print("Sum of IDs     :", total_id.item())

# 3. prove differentiability
total_id.backward()
print("Gradient present on X? ->", X.grad is not None
                                   and torch.all(torch.isfinite(X.grad)).item())
