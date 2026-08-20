"""Sample-wise teacher reliability estimators for UA-D2OFL.

All estimators share one signature and return weights of shape [B, R]
(batch x teachers), each row summing to 1. `teacher_probs` are T=1
probabilities [B, R, C]; `prototype_sims` are cosine similarities
between the CLIP image embedding of x and each teacher's class-c text
prototype [B, R]; `mask` is 1 for usable teachers, 0 for teachers that
must receive zero weight (e.g. missing prototype under the "mask"
policy) [B, R].
"""

import math
import torch

EPS = 1e-8


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor | None, tau: float) -> torch.Tensor:
    if mask is None:
        return torch.softmax(scores / tau, dim=1)
    # all-masked rows fall back to uniform over all teachers instead of NaN
    all_masked = mask.sum(dim=1, keepdim=True) == 0
    w = torch.softmax(scores.masked_fill((mask == 0) & ~all_masked, float("-inf")) / tau, dim=1)
    return torch.where(all_masked, torch.full_like(w, 1.0 / w.shape[1]), w)


def _standardize(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Per-sample z-score across teachers so heterogeneous signals share a scale.

    Needed because CLIP text-image cosine lives in a narrow band (modality
    gap) while normalized-entropy spans [0, 1].
    """
    if mask is None:
        mask = torch.ones_like(scores)
    n = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (scores * mask).sum(dim=1, keepdim=True) / n
    var = ((scores - mean) ** 2 * mask).sum(dim=1, keepdim=True) / n
    return (scores - mean) / (var.sqrt() + EPS)


def normalized_entropy_reliability(teacher_probs: torch.Tensor) -> torch.Tensor:
    """R_unc in [0, 1]: 1 - H(p)/log(C). Higher = more certain."""
    p = teacher_probs
    h = -(p * (p + EPS).log()).sum(dim=-1)
    return 1.0 - h / math.log(p.shape[-1])


def uniform_weights(teacher_probs, prototype_sims=None, mask=None, tau=1.0):
    scores = torch.zeros(teacher_probs.shape[:2], device=teacher_probs.device)
    return _masked_softmax(scores, mask, tau)


def confidence_weights(teacher_probs, prototype_sims=None, mask=None, tau=1.0):
    conf = teacher_probs.max(dim=-1).values
    return _masked_softmax(conf, mask, tau)


def entropy_weights(teacher_probs, prototype_sims=None, mask=None, tau=1.0):
    return _masked_softmax(normalized_entropy_reliability(teacher_probs), mask, tau)


def prototype_weights(teacher_probs, prototype_sims=None, mask=None, tau=1.0):
    assert prototype_sims is not None, "prototype_weights requires prototype_sims"
    return _masked_softmax(prototype_sims, mask, tau)


def joint_weights(teacher_probs, prototype_sims=None, mask=None, tau=1.0,
                  beta=0.5, standardize=True):
    """R = beta * R_unc + (1-beta) * R_sem, softmaxed over teachers."""
    r_unc = normalized_entropy_reliability(teacher_probs)
    if beta == 1.0 or prototype_sims is None:
        scores = r_unc if not standardize else _standardize(r_unc, mask)
    else:
        r_sem = prototype_sims
        if standardize:
            r_unc, r_sem = _standardize(r_unc, mask), _standardize(r_sem, mask)
        scores = beta * r_unc + (1.0 - beta) * r_sem
    return _masked_softmax(scores, mask, tau)


ESTIMATORS = {
    "uniform": uniform_weights,
    "confidence": confidence_weights,
    "entropy": entropy_weights,
    "prototype": prototype_weights,
    "joint": joint_weights,
}


def compute_weights(name: str, teacher_probs, prototype_sims=None, mask=None,
                    tau=1.0, **kwargs) -> torch.Tensor:
    return ESTIMATORS[name](teacher_probs, prototype_sims=prototype_sims,
                            mask=mask, tau=tau, **kwargs)
