"""Paper-scale phase 1 distillation: NICO++ Common 60 classes.

The full §11-style comparison at one setting (feature- or label-skew,
selected via UA_WORK), matched seeds, per-client eval breakdown.

Usage (on VM, after cache stage):
  python -m ua_d2ofl.experiments.paper_run --seed 0
  python -m ua_d2ofl.experiments.paper_run --report
"""

import argparse
import json
import os

import torch

from ua_d2ofl.data.dataset import ManifestDataset
from ua_d2ofl.metrics import evaluate
from ua_d2ofl.server.distill import (aggregate_cache, calibrate_cache,
                                     compute_all_weights, train_global)

WORK = os.environ.get("UA_WORK", "/content/work_f60")
EPOCHS = int(os.environ.get("UA_DISTILL_EPOCHS", "30"))
TAU, BETA, ALPHA, KD_T = 0.5, 0.5, 0.5, 3.0
RESULTS = f"{WORK}/results_paper.json"

CONFIGS = [
    ("uniform", {}),
    ("oracle", {}),
    ("confidence", {}),
    ("entropy", {}),
    ("prototype|blend0.5", {"missing_policy": "blend", "blend_lambda": 0.5}),
    ("joint|mask", {"missing_policy": "mask"}),
    ("joint|blend0.25", {"missing_policy": "blend", "blend_lambda": 0.25}),
    # calibration-aware KD: per-teacher temperature fitted on D_syn (idx 7-9)
    ("uniform|cal", {"_calibrate": "all"}),
    ("uniform|calsup", {"_calibrate": "support"}),
    ("joint|blend0.25|calsup", {"missing_policy": "blend", "blend_lambda": 0.25,
                                "_calibrate": "support"}),
    # reviewer controls (idx 10-13)
    ("sizeprop", {}),
    ("scalar-entropy", {}),
    ("labelce", {}),
    ("entropy|shuffled", {"shuffled": True}),
    # robust-aggregation controls (idx 14-16)
    ("median", {"_aggregate": "median"}),
    ("trimmed", {"_aggregate": "trimmed"}),
    ("agreement", {}),
    # lambda sweep for the blend policy (idx 17-19; 0.25 is idx 6)
    ("joint|blend0.1", {"missing_policy": "blend", "blend_lambda": 0.1}),
    ("joint|blend0.5", {"missing_policy": "blend", "blend_lambda": 0.5}),
    ("joint|blend0.75", {"missing_policy": "blend", "blend_lambda": 0.75}),
    # learned gate (IntactOFL-style), idx 20
    ("gate", {}),
]


def run_seed(seed: int, config_idx: int | None = None) -> None:
    with open(f"{WORK}/manifests.json", encoding="utf-8") as f:
        m = json.load(f)
    data_root = os.environ.get("UA_DATA", "/content/nico")
    for samples in m["test"].values():
        for s in samples:
            if not os.path.isabs(s["path"]):
                s["path"] = os.path.join(data_root, s["path"])
    blob = torch.load(f"{WORK}/cache.pt", map_location="cpu", weights_only=False)
    cache, syn = blob["cache"], blob["syn"]

    results = {}
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            results = json.load(f)
    todo = CONFIGS if config_idx is None else [CONFIGS[config_idx]]
    for name, kw in todo:
        key = f"{name}|s{seed}"
        if key in results:
            print(f"{key}: cached, skipping", flush=True)
            continue
        kw = dict(kw)
        cal = kw.pop("_calibrate", None)
        agg = kw.pop("_aggregate", None)
        cache_eff = cache
        if cal:
            cache_eff = calibrate_cache(cache, support_only=cal == "support")
        elif agg:
            cache_eff = aggregate_cache(cache, agg, KD_T)
        sizes = [len(m["train"][str(r)]) for r in range(len(m["train"]))]
        method = "uniform" if agg else name.split("|")[0]
        torch.manual_seed(seed)  # gate training is stochastic; others ignore it
        weights, diag = compute_all_weights(cache_eff, method,
                                            tau=TAU, beta=BETA,
                                            client_sizes=sizes, **kw)
        torch.manual_seed(seed)
        model = train_global(syn, cache_eff, weights, len(m["classes"]), epochs=EPOCHS,
                             alpha=ALPHA, kd_temperature=KD_T,
                             cosine=os.environ.get("UA_DISTILL_COS") == "1")
        per_client = {r: evaluate(model, ManifestDataset(m["test"][r]))
                      for r in m["test"]}
        overall = {k: sum(d[k] * d["n"] for d in per_client.values()) /
                      sum(d["n"] for d in per_client.values())
                   for k in ["acc", "ece", "nll", "brier"]}
        results[key] = {"overall": overall, "per_client": per_client,
                        "H_w": diag["summary"]["mean_weight_entropy"],
                        "corr_wc": diag["summary"]["corr_weight_correct"],
                        "missing_rate": diag["summary"]["missing_prototype_rate"]}
        with open(RESULTS, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"{key}: acc={overall['acc']:.4f} ece={overall['ece']:.4f} "
              f"H_w={results[key]['H_w']:.3f}", flush=True)


def report() -> None:
    with open(RESULTS, encoding="utf-8") as f:
        results = json.load(f)
    print(f"{'config':<22}{'acc mean±std':>16}{'ece mean±std':>16}")
    for name, _ in CONFIGS:
        runs = [v for k, v in results.items() if k.rsplit("|s", 1)[0] == name]
        if not runs:
            continue
        accs = torch.tensor([r["overall"]["acc"] for r in runs])
        eces = torch.tensor([r["overall"]["ece"] for r in runs])
        print(f"{name:<22}{accs.mean():>8.4f}±{accs.std():<6.4f}"
              f"{eces.mean():>8.4f}±{eces.std():<6.4f}(n={len(runs)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", type=int, default=None, help="index into CONFIGS")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        report()
    else:
        run_seed(a.seed if a.seed is not None else 0, a.config)
