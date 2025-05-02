import torch
import torchsort                               # pip install torchsort


def _pairwise_dists(X: torch.Tensor, X_is_dist: bool) -> torch.Tensor:
    """Squared-Euclidean pairwise distances (or pass-through if already provided)."""
    return X if X_is_dist else torch.cdist(X, X, p=2)


def _soft_sort(vec: torch.Tensor, tau: float | None) -> torch.Tensor:
    """
    Exact sort when `tau is None`, otherwise differentiable SoftSort.
    torchsort.soft_sort expects an extra batch-dim so we unsqueeze / squeeze.
    """
    if tau is None:
        return torch.sort(vec, dim=-1)[0]          # exact, piece-wise-const grads
    return torchsort.soft_sort(vec.unsqueeze(0),
                               regularization_strength=tau).squeeze(0)


# --------------------------------------------------------------------------- #
# 1)  Extreme-value Maximum-Likelihood (Hill) estimator 
# --------------------------------------------------------------------------- #
def estimate_id_torch_mle(
    X: torch.Tensor,
    k: int = 20,
    tau: float | None = None,
    X_is_dist: bool = False,
) -> torch.Tensor:
    """
    Differentiable Hill / MLE estimator of (global) LID.

    Parameters
    ----------
    X         : (N, p) samples or (N, N) distance matrix
    k         : #nearest-neighbours used in the estimate (k = 2)
    tau       : SoftSort temperature.  None ? exact sort
    X_is_dist : set True if `X` already holds pairwise distances

    Returns
    -------
    id_hat    : scalar *tensor*
    """
    if k < 2:
        raise ValueError("k must be = 2")

    dist = _pairwise_dists(X, X_is_dist)
    d_sorted = _soft_sort(dist, tau)               # (N, N)

    r = d_sorted[:, 1:k+1]                         # drop zero self-distance
    r_k = r[:, -1:].detach() * 1                   # (N, 1) broadcast later

    # Hill statistic per-sample (no minus sign because we invert positive logs)
    t = torch.log(r_k / r)                         # positive
    local_id = (k / t.sum(dim=1))                  # (N,)

    return local_id.mean()                         # global ID (tensor)



