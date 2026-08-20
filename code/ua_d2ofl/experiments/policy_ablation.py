"""Missing-prototype policy ablation on the label-skew smoke setting.

Motivated by the observed single-teacher collapse: under disjoint label skew
the "mask" policy reduces prototype/joint to owner-only distillation, which
loses the specialist-ensemble dark knowledge and underperforms uniform.
Compares mask / ignore / fallback / blend(lambda) across matched seeds.

Usage (on VM, after cache stage):
  python -m ua_d2ofl.experiments.policy_ablation --seed 0
  python -m ua_d2ofl.experiments.policy_ablation --report
"""

import argparse
import json
import os

import torch

from ua_d2ofl.data.dataset import ManifestDataset
from ua_d2ofl.metrics import evaluate
from ua_d2ofl.server.distill import compute_all_weights, train_global

WORK = os.environ.get("UA_WORK", "/content/work_label")
EPOCHS = 30
TAU, BETA, ALPHA, KD_T = 0.5, 0.5, 0.5, 3.0
RESULTS = f"{WORK}/results_policies.json"

CONFIGS = [
    ("uniform", {}),
    ("entropy", {}),
    ("prototype|mask", {"missing_policy": "mask"}),
    ("prototype|ignore", {"missing_policy": "ignore"}),
    ("prototype|fallback", {"missing_policy": "fallback"}),
    ("prototype|blend0.5", {"missing_policy": "blend", "blend_lambda": 0.5}),
    ("joint|mask", {"missing_policy": "mask"}),
    ("joint|ignore", {"missing_policy": "ignore"}),
    ("joint|fallback", {"missing_policy": "fallback"}),
    ("joint|blend0.25", {"missing_policy": "blend", "blend_lambda": 0.25}),
    ("joint|blend0.5", {"missing_policy": "blend", "blend_lambda": 0.5}),
    ("joint|blend0.75", {"missing_policy": "blend", "blend_lambda": 0.75}),
]


def _load():
    with open(f"{WORK}/manifests.json", encoding="utf-8") as f:
        m = json.load(f)
    data_root = os.environ.get("UA_DATA", "/content/nico")
    for samples in m["test"].values():
        for s in samples:
            if not os.path.isabs(s["path"]):
                s["path"] = os.path.join(data_root, s["path"])
    blob = torch.load(f"{WORK}/cache.pt", map_location="cpu", weights_only=False)
    return m, blob["cache"], blob["syn"]


def run_seed(seed: int) -> None:
    m, cache, syn = _load()
    results = {}
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            results = json.load(f)
    for name, kw in CONFIGS:
        key = f"{name}|s{seed}"
        if key in results:
            continue
        method = name.split("|")[0]
        weights, diag = compute_all_weights(cache, method, tau=TAU, beta=BETA, **kw)
        torch.manual_seed(seed)
        model = train_global(syn, cache, weights, len(m["classes"]), epochs=EPOCHS,
                             alpha=ALPHA, kd_temperature=KD_T)
        per_client = {r: evaluate(model, ManifestDataset(m["test"][r]))
                      for r in m["test"]}
        overall = {k: sum(d[k] * d["n"] for d in per_client.values()) /
                      sum(d["n"] for d in per_client.values())
                   for k in ["acc", "ece", "nll", "brier"]}
        results[key] = {"overall": overall, "per_client": per_client,
                        "H_w": diag["summary"]["mean_weight_entropy"],
                        "corr_wc": diag["summary"]["corr_weight_correct"]}
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"{key}: acc={overall['acc']:.4f} ece={overall['ece']:.4f} "
              f"H_w={results[key]['H_w']:.3f}", flush=True)


def report() -> None:
    with open(RESULTS, encoding="utf-8") as f:
        results = json.load(f)
    print(f"{'config':<22}{'acc mean±std':>16}{'ece mean±std':>16}{'H_w':>7}")
    for name, _ in CONFIGS:
        runs = [v for k, v in results.items() if k.rsplit("|s", 1)[0] == name]
        if not runs:
            continue
        accs = torch.tensor([r["overall"]["acc"] for r in runs])
        eces = torch.tensor([r["overall"]["ece"] for r in runs])
        print(f"{name:<22}{accs.mean():>8.4f}±{accs.std():<6.4f}"
              f"{eces.mean():>8.4f}±{eces.std():<6.4f}{runs[0]['H_w']:>7.3f}"
              f"  (n={len(runs)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        report()
    else:
        run_seed(a.seed if a.seed is not None else 0)
