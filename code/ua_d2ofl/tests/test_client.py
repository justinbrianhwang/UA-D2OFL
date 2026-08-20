"""Client-pipeline sanity tests (local, no foundation models needed).
Run: python -m ua_d2ofl.tests.test_client
"""

import os
import tempfile

import torch
from PIL import Image

from ua_d2ofl.client.filtering import filter_and_average, mahalanobis_keep_mask
from ua_d2ofl.data.partition import build_manifest, dirichlet_skew, feature_skew, label_skew

torch.manual_seed(0)


def test_filter_drops_planted_outlier():
    z = torch.randn(30, 768) * 0.05 + 1.0
    z[7] += 10.0  # planted outlier
    keep = mahalanobis_keep_mask(z, k=3.0)
    assert not keep[7]
    assert keep.sum() >= 25  # inliers mostly survive


def test_filter_small_n_keeps_all():
    assert mahalanobis_keep_mask(torch.randn(3, 768)).all()


def test_filter_and_average_consistent():
    hidden, pooled = torch.randn(20, 77, 8), torch.randn(20, 8)
    pooled[3] += 50.0
    protos, stats = filter_and_average({"cond": hidden, "pooled": pooled})
    assert protos["cond"].shape == (77, 8) and protos["pooled"].shape == (8,)
    assert stats["n_kept"] + stats["n_dropped"] == 20
    assert stats["n_dropped"] >= 1  # planted outlier dropped from BOTH reps


def _fake_tree(root, domains=("autumn", "rock"), classes=("bear", "cat"), n=4):
    for d in domains:
        for c in classes:
            os.makedirs(os.path.join(root, d, c), exist_ok=True)
            for i in range(n):
                Image.new("RGB", (32, 32), (i * 40, 0, 0)).save(
                    os.path.join(root, d, c, f"{i}.png"))


def test_partitions():
    with tempfile.TemporaryDirectory() as root:
        _fake_tree(root)
        manifest, classes = build_manifest(root)
        assert len(manifest) == 2 * 2 * 4 and classes == ["bear", "cat"]

        fs = feature_skew(manifest)
        assert len(fs) == 2
        for cid, samples in fs.items():
            assert len({s["domain"] for s in samples}) == 1

        ls = label_skew(manifest, num_clients=2, seed=0)
        seen = [{s["label"] for s in v} for v in ls.values()]
        assert seen[0].isdisjoint(seen[1])
        assert sum(len(v) for v in ls.values()) == len(manifest)

        d1 = dirichlet_skew(manifest, 3, alpha=0.5, seed=1)
        d2 = dirichlet_skew(manifest, 3, alpha=0.5, seed=1)
        assert sum(len(v) for v in d1.values()) == len(manifest)
        assert all([s["path"] for s in d1[i]] == [s["path"] for s in d2[i]] for i in d1)


def test_teacher_train_smoke():
    from ua_d2ofl.client.train_local import train_teacher
    with tempfile.TemporaryDirectory() as root:
        _fake_tree(root, n=3)
        manifest, _ = build_manifest(root)
        model = train_teacher(manifest, num_classes=2, epochs=1, batch_size=4,
                              num_workers=0, pretrained=False)
        out = model(torch.randn(2, 3, 224, 224).to(next(model.parameters()).device))
        assert out.shape == (2, 2)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
    print("all tests passed")
