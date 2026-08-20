"""Reviewer control (F5): uniform distillation + post-hoc student-side
temperature scaling, with T fitted on the synthetic transfer set (the only
labeled data the server has). If this matches the weighted schemes' ECE, the
calibration benefit of reliability weighting is not unique.

Usage (on VM, after cache stage): python -m ua_d2ofl.experiments.student_ts --seed 0
Writes key "uniform+studentTS|s{seed}" into the same results_paper.json.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from ua_d2ofl.data.dataset import ManifestDataset, TEST_TF
from ua_d2ofl.metrics.calibration import brier, ece, nll
from ua_d2ofl.server.distill import compute_all_weights, train_global

WORK = os.environ.get("UA_WORK", "/content/work_f60")
EPOCHS = int(os.environ.get("UA_DISTILL_EPOCHS", "30"))
TAU, BETA, ALPHA, KD_T = 0.5, 0.5, 0.5, 3.0
RESULTS = f"{WORK}/results_paper.json"


@torch.no_grad()
def _logits(model, samples, device):
    model.eval()
    loader = DataLoader(ManifestDataset(samples, TEST_TF), batch_size=256,
                        num_workers=2)
    outs, ys = [], []
    for x, y in loader:
        outs.append(model(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def main(seed: int) -> None:
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
    key = f"uniform+studentTS|s{seed}"
    if key in results:
        print(f"{key}: cached, skipping", flush=True)
        return

    weights, _ = compute_all_weights(cache, "uniform", tau=TAU, beta=BETA)
    torch.manual_seed(seed)
    model = train_global(syn, cache, weights, len(m["classes"]), epochs=EPOCHS,
                         alpha=ALPHA, kd_temperature=KD_T,
                         cosine=os.environ.get("UA_DISTILL_COS") == "1")
    device = next(model.parameters()).device

    syn_logits, syn_labels = _logits(model, syn, device)
    t_fit = fit_temperature(syn_logits, syn_labels)
    print(f"fitted student temperature on D_syn: {t_fit:.3f}", flush=True)

    alltest = [s for v in m["test"].values() for s in v]
    te_logits, te_labels = _logits(model, alltest, device)
    out = {}
    for tag, t in [("raw", 1.0), ("ts", t_fit)]:
        probs = torch.softmax(te_logits / t, dim=1)
        out[tag] = {"acc": (probs.argmax(1) == te_labels).float().mean().item(),
                    "ece": ece(probs, te_labels), "nll": nll(probs, te_labels),
                    "brier": brier(probs, te_labels), "n": len(te_labels)}
    results[key] = {"overall": out["ts"], "raw": out["raw"],
                    "fitted_T": t_fit, "H_w": None, "corr_wc": None,
                    "missing_rate": None}
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"{key}: acc={out['ts']['acc']:.4f} ece={out['ts']['ece']:.4f} "
          f"(raw ece={out['raw']['ece']:.4f}) T={t_fit:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args().seed)
