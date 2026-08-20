"""Heterogeneity-severity experiment (H2): Dirichlet(alpha) label partitions.

Runs the four story methods (uniform baseline, entropy, joint|mask,
joint|blend0.25) on one Dirichlet work dir; the runner sweeps alpha via
UA_WORK. Test evaluation uses the shared global test pool (m["test"]["all"]).

Usage (on VM): python -m ua_d2ofl.experiments.severity --seed 0
"""

import argparse
import json
import os

import torch

from ua_d2ofl.data.dataset import ManifestDataset
from ua_d2ofl.metrics import evaluate
from ua_d2ofl.server.distill import compute_all_weights, train_global

WORK = os.environ.get("UA_WORK", "/content/work_d")
EPOCHS = 30
TAU, BETA, ALPHA, KD_T = 0.5, 0.5, 0.5, 3.0
RESULTS = f"{WORK}/results_severity.json"

CONFIGS = [
    ("uniform", {}),
    ("entropy", {}),
    ("joint|mask", {"missing_policy": "mask"}),
    ("joint|blend0.25", {"missing_policy": "blend", "blend_lambda": 0.25}),
]


def run_seed(seed: int) -> None:
    with open(f"{WORK}/manifests.json", encoding="utf-8") as f:
        m = json.load(f)
    data_root = os.environ.get("UA_DATA", "/content/nico")
    for s in m["test"]["all"]:
        if not os.path.isabs(s["path"]):
            s["path"] = os.path.join(data_root, s["path"])
    blob = torch.load(f"{WORK}/cache.pt", map_location="cpu", weights_only=False)
    cache, syn = blob["cache"], blob["syn"]
    test_set = ManifestDataset(m["test"]["all"])

    results = {}
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            results = json.load(f)
    for name, kw in CONFIGS:
        key = f"{name}|s{seed}"
        if key in results:
            continue
        weights, diag = compute_all_weights(cache, name.split("|")[0],
                                            tau=TAU, beta=BETA, **kw)
        torch.manual_seed(seed)
        model = train_global(syn, cache, weights, len(m["classes"]), epochs=EPOCHS,
                             alpha=ALPHA, kd_temperature=KD_T)
        overall = evaluate(model, test_set)
        results[key] = {"overall": overall,
                        "H_w": diag["summary"]["mean_weight_entropy"],
                        "missing_rate": diag["summary"]["missing_prototype_rate"],
                        "mode": m.get("mode", "?")}
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"{key}: acc={overall['acc']:.4f} ece={overall['ece']:.4f} "
              f"H_w={results[key]['H_w']:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run_seed(a.seed)
