"""Mahalanobis-distance encoding filter (D2OFL Eq. 11-15).

Given a category's embedding set Z [N, D], drop outliers whose squared
Mahalanobis distance exceeds mean + k * std of the distances.
"""

import torch


def mahalanobis_keep_mask(z: torch.Tensor, k: float = 3.0, shrinkage: float = 0.1) -> torch.Tensor:
    """Return a bool mask [N] of embeddings to KEEP.

    ponytail: N (captions per category, ~30) << D (768) makes the sample
    covariance singular, which the paper doesn't address; we shrink toward
    the scaled identity (Sigma + shrinkage * mean_var * I). Tune shrinkage
    if filtering looks degenerate.
    """
    n, d = z.shape
    if n < 4:  # too few points for meaningful statistics
        return torch.ones(n, dtype=torch.bool, device=z.device)
    z = z.double()
    mu = z.mean(dim=0)
    zc = z - mu
    cov = (zc.T @ zc) / n
    mean_var = cov.diagonal().mean()
    cov += shrinkage * mean_var * torch.eye(d, dtype=cov.dtype, device=cov.device)
    dist = (zc * torch.linalg.solve(cov, zc.T).T).sum(dim=1)  # squared Mahalanobis
    delta = dist.mean() + k * dist.std(unbiased=False)
    return dist <= delta


def filter_and_average(embeddings: dict[str, torch.Tensor], k: float = 3.0,
                       filter_on: str = "pooled") -> tuple[dict[str, torch.Tensor], dict]:
    """Filter a category's embedding sets and average the survivors.

    `embeddings` maps representation name -> [N, ...] tensors sharing sample
    order (e.g. {"pooled": [N,768], "cond": [N,77,768]}). The keep decision is
    computed once on `filter_on` (pooled CLIP embeddings; the 77x768
    conditioning states are too big for covariance estimation) and applied to
    every representation. Returns ({name: mean_of_kept}, stats).
    """
    keep = mahalanobis_keep_mask(embeddings[filter_on].flatten(1), k=k)
    protos = {name: e[keep].mean(dim=0) for name, e in embeddings.items()}
    stats = {"n_total": keep.numel(), "n_kept": int(keep.sum()),
             "n_dropped": int((~keep).sum())}
    return protos, stats
