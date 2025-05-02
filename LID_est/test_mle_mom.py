from LID_estimator import estimate_id_torch_soft
from MLE_MoM import estimate_id_torch_mle
import torch



def make_linear_subspace_data(N=4000, d_true=5, p=30, device="cpu"):
    """
    Sample N points from a d_true-D Gaussian and embed in R^p via
    a random orthonormal matrix (linear preserves LID).
    """
    X_low = torch.randn(N, d_true, device=device)
    W = torch.linalg.qr(torch.randn(p, d_true, device=device))[0]   
    return X_low @ W.T                                              



def test_all(N=4000, d_true=5, p=30, ks=(10, 20, 40, 80), tau=None, plot=True):
    X = make_linear_subspace_data(N, d_true, p).requires_grad_(True)

    print(f"\nTrue intrinsic dimension = {d_true}\n")
    print("  k   Two-NN   MLE(Hill)")
    print("-" * 34)

    two_nn = estimate_id_torch_soft(X, tau=1.0)      
    results = {"TwoNN": [], "MLE": []}

    for k in ks:
        id_mle = estimate_id_torch_mle(X, k=k, tau=tau)
        results["MLE"].append(id_mle.item())
        print(f"{k:3d}  {two_nn.item():7.2f}   {id_mle.item():9.2f}")



if __name__ == "__main__":
    test_all()