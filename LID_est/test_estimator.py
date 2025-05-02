"""
Test-drive for estimate_id_torch_soft
------------------------------------
$ python check_id_soft.py
"""

import torch
import torchsort                       # pip install torchsort
# ────────────────────────────────────────────────────────────────────────────────
# Copy–paste (or `import`) your implementation of estimate_id_torch_soft here
from LID_estimator import estimate_id_torch_soft
# from estimate_id_soft import estimate_id_torch_soft   # <- adjust if needed
# ────────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------------
    # 1. create a synthetic dataset: 5-D linear subspace in 32-D + small noise
    # ---------------------------------------------------------------------
    N           = 1024
    true_dim    = 5
    ambient_dim = 32

    A       = torch.randn(true_dim, ambient_dim, device=device)
    coeff   = torch.randn(N, true_dim, device=device)
    X_clean = coeff @ A                               # lives in 5-D sub-space
    X_noise = 0.05 * torch.randn_like(X_clean)        # small isotropic noise

    X = (X_clean + X_noise).requires_grad_(True)      # pretend it came from a net

    # ---------------------------------------------------------------------
    # 2. estimate intrinsic dimension with differentiable Two-NN + SoftSort
    # ---------------------------------------------------------------------
    id_est = estimate_id_torch_soft(X, tau=0.5)       # scalar *tensor*
    print(f"Estimated ID ≈ {id_est.item():.2f} (ground-truth 5)")

    # ---------------------------------------------------------------------
    # 3. back-prop check: slope is our “loss”, make sure grads propagate
    # ---------------------------------------------------------------------
    id_est.backward()                                 # should populate X.grad
    grad_ok = X.grad is not None and torch.all(torch.isfinite(X.grad)).item()
    print("Gradient present and finite:", grad_ok)    # → True

if __name__ == "__main__":
    main()
