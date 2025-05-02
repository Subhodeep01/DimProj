import torch
import torchsort                                # pip install torchsort
import matplotlib.pyplot as plt


def estimate_id_torch_soft(
    X: torch.Tensor,
    tau: float = 1.0,
    plot: bool = False,
    X_is_dist: bool = False,
) -> torch.Tensor:
    """
    Differentiable Two-NN intrinsic-dimension estimator (Facco et al., 2017)
    that uses `torchsort.soft_sort` so the *global* slope is back-propable.

    Parameters
    ----------
    X        : (N, p) data tensor  **or**  (N, N) distance matrix.
               Must come from an upstream module with `requires_grad=True`
               if you want gradients.
    tau      : temperature (regularisation strength) for SoftSort.
               Smaller τ  -> closer to true sort but sharper gradients.
    plot     : optionally show the log–log fit (detached, no gradient impact).
    X_is_dist: set True if `X` is already a pair-wise distance matrix.

    Returns
    -------
    slope    : 0-D *tensor*  (leave it as a tensor if you plan to back-prop;
               call `.item()` only for logging).
    """

    N = X.shape[0]

    # 1. pair-wise distances -----------------------------------------------
    dist = X if X_is_dist else torch.cdist(X, X, p=2)

    # 2. μ_i = r₂ / r₁  (exclude the zero self-distance) -------------------
    dist_sorted, _ = torch.sort(dist, dim=1)          # exact; differentiable
    eps = 1e-8                                         # NEW
    mu = dist_sorted[:, 2] / (dist_sorted[:, 1] + eps) # avoid 0 denominator

    # 3. Soft sort to obtain a differentiable "sorted" vector --------------
    # torchsort expects an extra batch dim, so use unsqueeze / squeeze
    mu_sorted = torchsort.soft_sort(
        mu.unsqueeze(0), regularization_strength=tau
    ).squeeze(0)                                      # (N,)

    # empirical CDF positions (constants, no grad needed)
    F_emp = torch.arange(N, dtype=mu.dtype, device=mu.device) / N

    # 4. slope of the log–log plot through the origin ----------------------
    x = torch.log(torch.clamp(mu_sorted, min=eps))     # avoid log(0)
    y = -torch.log1p(-F_emp)                          # −log(1 – F)

    slope = (x * y).sum() / (x * x).sum()            # scalar *tensor*

    # 5. optional diagnostic plot (detached copy, so gradients unaffected) --
    if plot:
        with torch.no_grad():
            plt.scatter(x.cpu(), y.cpu(), s=8, c="red", label="soft-sorted data")
            plt.plot(x.cpu(), (slope.detach() * x).cpu(),
                     c="black", lw=2, label=f"linear fit\nID ≈ {slope.item():.3f}")
            plt.xlabel(r"$\\log(\\mu_i)$")
            plt.ylabel(r"$-\\log\\bigl(1 - F_{\\mathrm{emp}}(\\mu_i)\\bigr)$")
            plt.title("Differentiable Two-NN Intrinsic Dimensionality")
            plt.legend()
            plt.show()

    return slope
