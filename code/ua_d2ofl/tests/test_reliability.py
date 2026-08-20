"""Sanity tests from research prompt section 23. Run: python -m ua_d2ofl.tests.test_reliability"""

import math
import torch

from ua_d2ofl.distillation import compute_weights, weighted_kd_loss
from ua_d2ofl.distillation.reliability import normalized_entropy_reliability
from ua_d2ofl.distillation.kd_loss import ensemble_target

torch.manual_seed(0)
B, R, C = 8, 6, 10
logits = torch.randn(B, R, C) * 3
probs = torch.softmax(logits, dim=-1)
sims = torch.rand(B, R) * 0.1 + 0.2  # narrow band, mimics CLIP modality gap
labels = torch.randint(0, C, (B,))


def test_rows_sum_to_one():
    for name in ["uniform", "confidence", "entropy", "prototype", "joint"]:
        w = compute_weights(name, probs, prototype_sims=sims, tau=0.5)
        assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-5), name
        assert (w >= 0).all(), name


def test_uniform_reproduces_d2ofl():
    w = compute_weights("uniform", probs)
    assert torch.allclose(w, torch.full((B, R), 1 / R), atol=1e-6)
    tgt = ensemble_target(logits, w, temperature=3.0)
    ref = torch.softmax(logits / 3.0, dim=-1).mean(dim=1)
    assert torch.allclose(tgt, ref, atol=1e-6)


def test_masked_teacher_zero_weight():
    mask = torch.ones(B, R)
    mask[:, 0] = 0
    for name in ["uniform", "entropy", "prototype", "joint"]:
        w = compute_weights(name, probs, prototype_sims=sims, mask=mask, tau=0.5)
        assert torch.allclose(w[:, 0], torch.zeros(B), atol=1e-6), name
        assert torch.allclose(w.sum(1), torch.ones(B), atol=1e-5), name
    # all-masked row falls back to uniform, never NaN
    w = compute_weights("entropy", probs, mask=torch.zeros(B, R))
    assert torch.allclose(w, torch.full((B, R), 1 / R), atol=1e-6)


def test_entropy_bounded():
    r = normalized_entropy_reliability(probs)
    assert (r >= -1e-5).all() and (r <= 1 + 1e-5).all()
    one_hot = torch.zeros(1, 1, C); one_hot[..., 0] = 1
    assert normalized_entropy_reliability(one_hot).item() > 0.999
    unif = torch.full((1, 1, C), 1 / C)
    assert abs(normalized_entropy_reliability(unif).item()) < 1e-5


def test_joint_beta_extremes():
    w_ent = compute_weights("joint", probs, prototype_sims=sims, tau=0.5, beta=1.0)
    w_ent_only = compute_weights("joint", probs, prototype_sims=None, tau=0.5, beta=1.0)
    assert torch.allclose(w_ent, w_ent_only, atol=1e-6)
    w_sem = compute_weights("joint", probs, prototype_sims=sims, tau=0.5, beta=0.0)
    w_proto_std = compute_weights("joint", probs, prototype_sims=sims, tau=0.5, beta=0.0, standardize=True)
    assert torch.allclose(w_sem, w_proto_std, atol=1e-6)


def test_tau_sharpness():
    w_flat = compute_weights("entropy", probs, tau=100.0)
    w_sharp = compute_weights("entropy", probs, tau=0.05)
    assert torch.allclose(w_flat, torch.full((B, R), 1 / R), atol=1e-3)
    assert w_sharp.max(1).values.mean() > w_flat.max(1).values.mean()
    # entropy of sharp weights << entropy of flat weights
    h = lambda w: -(w * (w + 1e-8).log()).sum(1).mean()
    assert h(w_sharp) < h(w_flat) < math.log(R) + 1e-3


def test_gradients_only_student():
    student_logits = torch.randn(B, C, requires_grad=True)
    t_logits = logits.clone().requires_grad_(True)
    w = compute_weights("joint", torch.softmax(t_logits, -1), prototype_sims=sims, tau=0.5)
    loss = weighted_kd_loss(student_logits, t_logits, labels, w, alpha=0.5, temperature=3.0)
    loss.backward()
    assert student_logits.grad is not None and student_logits.grad.abs().sum() > 0
    assert t_logits.grad is None or t_logits.grad.abs().sum() == 0


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all tests passed")
