"""Weighted multi-teacher distillation loss (D2OFL Eq. 19, generalized).

L = alpha * CE(student, y) + (1-alpha) * T^2 * KL(p_bar_T || p_student_T)
with p_bar_T(.|x) = sum_r w_r(x) * softmax(f_r(x)/T).

Uniform w reproduces D2OFL-Uniform exactly. Teacher logits are detached;
gradients flow only through the student.
"""

import torch
import torch.nn.functional as F

EPS = 1e-8


def ensemble_target(teacher_logits: torch.Tensor, weights: torch.Tensor,
                    temperature: float) -> torch.Tensor:
    """teacher_logits [B, R, C], weights [B, R] -> p_bar_T [B, C]."""
    p_t = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    return (weights.detach().unsqueeze(-1) * p_t).sum(dim=1)


def weighted_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     labels: torch.Tensor, weights: torch.Tensor,
                     alpha: float = 0.5, temperature: float = 3.0) -> torch.Tensor:
    p_bar = ensemble_target(teacher_logits, weights, temperature)
    log_p_student_t = F.log_softmax(student_logits / temperature, dim=-1)
    kl = (p_bar * ((p_bar + EPS).log() - log_p_student_t)).sum(dim=-1).mean()
    ce = F.cross_entropy(student_logits, labels)
    return alpha * ce + (1.0 - alpha) * (temperature ** 2) * kl
