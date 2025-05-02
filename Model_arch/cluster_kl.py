import torch
import torch.nn.functional as F

_TINY = 1e-12        # numerical-stability constant


def cal_latent(hidden: torch.Tensor, alpha: float):
    """
    Student-t affinity (DEC/SCVI style) for a batch of latent vectors.

    Parameters
    ----------
    hidden : (N, D) tensor – latent representations
    alpha  : positive scalar – degrees-of-freedom parameter

    Returns
    -------
    num      : (N, N) tensor, unnormalised affinities (diagonal kept)
    latent_p : (N, N) tensor, row-normalised affinities with the diagonal removed
               (add the diagonal back later if you need it, exactly as the TF code does)
    """
    # pair-wise squared Euclidean distance matrix
    sum_y = (hidden ** 2).sum(dim=1, keepdim=True)           # (N, 1)
    dist  = sum_y + sum_y.T - 2.0 * hidden @ hidden.T        # (N, N)

    num = torch.pow(1.0 + dist / alpha, -0.5 * (alpha + 1.0))   # Student-t kernel, diag ≠ 0

    # remove diagonal for the row-normalised P_ij
    zerodiag_num = num - torch.diag_embed(torch.diagonal(num))   # diag→0, keeps grads
    row_sum      = zerodiag_num.sum(dim=1, keepdim=True).clamp_min(_TINY)
    latent_p     = zerodiag_num / row_sum                        # soft-probabilities
   
    
    return num, latent_p


def target_dis(latent_p: torch.Tensor):
    """
    “Target” distribution used in DEC / IDEC:
        q_ij = p_ij² / Σ_j p_ij   , then row-normalised.
    """
    weight = latent_p ** 2
    weight = weight / latent_p.sum(dim=1, keepdim=True).clamp_min(_TINY)
    latent_q = weight / weight.sum(dim=1, keepdim=True).clamp_min(_TINY)
    return latent_q


def cal_dist(hidden: torch.Tensor, clusters: torch.Tensor):
    """
    K-means-style distance & distance-weighted loss used in IDEC.

    Parameters
    ----------
    hidden   : (N, D)   – latent points
    clusters : (K, D)   – cluster centroids (learnable or from K-means)

    Returns
    -------
    dist1 : (N, K) – squared distances to each centroid
    dist2 : (N, K) – distance weighted by sharpened soft assignment q_ik
    """
    # squared Euclidean distance to each centroid
    dist1 = ((hidden.unsqueeze(1) - clusters) ** 2).sum(dim=2)   # (N, K)

    # soft assignment (RBF-like), shifted so the closest centroid has exp(0)=1
    temp_dist1 = dist1 - dist1.min(dim=1, keepdim=True)[0]       # rowwise min → 0
    q = torch.exp(-temp_dist1)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(_TINY)

    # sharpen and re-normalise 
    q = q.pow(2)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(_TINY)

    dist2 = dist1 * q
    return dist1, dist2
