import torch

def structured_pca(A: torch.Tensor, dim: int = 3) -> torch.Tensor:
    """
    Computes PCA for A and returns a embedding of the low rank approximation of A.
    """
    U, S, V = torch.pca_lowrank(A.flatten(0, -2), dim)
    return U.unflatten(0, A.shape[:-1])
