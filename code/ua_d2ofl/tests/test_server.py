"""Server-pipeline sanity tests (local, no foundation models needed).
Run: python -m ua_d2ofl.tests.test_server
"""

import math
import os
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image

from ua_d2ofl.metrics import brier, ece, evaluate, nll
from ua_d2ofl.server.distill import compute_all_weights, train_global
from ua_d2ofl.server.precompute import prototype_sims
from ua_d2ofl.data.dataset import ManifestDataset, TEST_TF

torch.manual_seed(0)
N, R, C = 24, 3, 4


def _fake_cache():
    labels = torch.randint(0, C, (N,))
    logits = torch.randn(N, R, C)
    logits[:, 0].scatter_(1, labels.unsqueeze(1), 6.0)  # teacher 0 is reliable
    avail = torch.ones(N, R)
    avail[labels % 2 == 0, 2] = 0  # teacher 2 misses even classes
    sims = torch.rand(N, R) * 0.1 + 0.2
    sims[:, 0] += 0.05
    return {"teacher_logits": logits, "image_embeds": torch.randn(N, 8),
            "sims": sims, "available": avail, "labels": labels}


def test_metrics_perfect_predictions():
    labels = torch.arange(4)
    probs = F.one_hot(labels, 4).float().clamp(1e-6, 1 - 1e-6)
    probs = probs / probs.sum(1, keepdim=True)
    assert ece(probs, labels) < 1e-4
    assert brier(probs, labels) < 1e-4
    assert nll(probs, labels) < 1e-4


def test_metrics_known_values():
    probs = torch.tensor([[0.7, 0.3], [0.7, 0.3]])
    labels = torch.tensor([0, 1])  # one right, one wrong at conf 0.7
    assert abs(ece(probs, labels) - 0.2) < 1e-6  # |acc 0.5 - conf 0.7|
    assert abs(nll(probs, labels) - (-(math.log(0.7) + math.log(0.3)) / 2)) < 1e-5
    assert abs(brier(probs, labels) - ((0.09 + 0.09 + 0.49 + 0.49) / 2)) < 1e-5


def test_prototype_sims_masks_missing():
    z = torch.randn(6, 8)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "p0.pt")  # client 0 has classes 0,1 only
        torch.save({0: {"pooled": torch.randn(8)}, 1: {"pooled": torch.randn(8)}}, p0)
        p1 = os.path.join(d, "p1.pt")  # client 1 has all three
        torch.save({lb: {"pooled": torch.randn(8)} for lb in range(3)}, p1)
        sims, avail = prototype_sims(z, labels, {0: p0, 1: p1})
    assert avail[:4, 0].all() and not avail[4:, 0].any()
    assert avail[:, 1].all()
    assert (sims[4:, 0] == 0).all()
    assert sims.abs().max() <= 1 + 1e-5  # cosine range


def test_weights_all_methods():
    cache = _fake_cache()
    for method in ["uniform", "confidence", "entropy", "prototype", "joint", "oracle"]:
        w, diag = compute_all_weights(cache, method, tau=0.5)
        assert w.shape == (N, R)
        assert torch.allclose(w.sum(1), torch.ones(N), atol=1e-5), method
        assert diag["weight_entropy"].max() <= math.log(R) + 1e-4
        assert "corr_weight_correct" in diag["summary"]
    w_u, _ = compute_all_weights(cache, "uniform")
    assert torch.allclose(w_u, torch.full((N, R), 1 / R), atol=1e-6)
    # oracle: zero weight where prototype missing, uniform over the rest
    w_o, _ = compute_all_weights(cache, "oracle")
    missing = cache["available"] == 0
    assert torch.allclose(w_o[missing], torch.zeros(missing.sum()), atol=1e-6)
    # reliable teacher 0 should out-weigh others under entropy weighting
    w_e, diag_e = compute_all_weights(cache, "entropy", tau=0.25)
    assert w_e[:, 0].mean() > 1 / R
    assert diag_e["summary"]["corr_weight_correct"] > 0


def test_train_global_end_to_end():
    cache = _fake_cache()
    with tempfile.TemporaryDirectory() as d:
        samples = []
        for i in range(N):
            path = os.path.join(d, f"{i}.png")
            Image.new("RGB", (32, 32), (i * 10 % 255, 0, 0)).save(path)
            samples.append({"path": path, "label": int(cache["labels"][i]), "client": 0})
        for method in ["uniform", "joint"]:
            w, _ = compute_all_weights(cache, method, tau=0.5)
            model = train_global(samples, cache, w, num_classes=C, epochs=1,
                                 batch_size=8, pretrained=False, num_workers=0)
            metrics = evaluate(model, ManifestDataset(samples, TEST_TF), batch_size=8)
            assert set(metrics) == {"acc", "ece", "nll", "brier", "n"}
            assert metrics["n"] == N


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all tests passed")
