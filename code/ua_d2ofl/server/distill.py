"""Global (student) distillation training — D2OFL Eq. 19 with pluggable
teacher weighting.

Weights depend only on the precomputed cache (frozen teachers, fixed
D_syn), never on the student, so they are computed once per method up
front; training then only runs the student forward/backward. Methods:
  uniform     D2OFL-Uniform (paper baseline, exact)
  confidence  UA-D2OFL-Confidence
  entropy     UA-D2OFL-Entropy
  prototype   UA-D2OFL-Prototype
  joint       UA-D2OFL-Joint
  oracle      uniform over teachers that HAVE a prototype for label(x) —
              the trivial class-mask baseline any UA variant must beat
              under label skew.
"""

import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ua_d2ofl.data.dataset import ManifestDataset, TEST_TF
from ua_d2ofl.distillation import compute_weights, weighted_kd_loss
from ua_d2ofl.distillation.reliability import (_standardize,
                                               normalized_entropy_reliability)
from ua_d2ofl.models import initialize_model

EPS = 1e-8


def fit_teacher_temperatures(cache: dict, support_only: bool = False) -> torch.Tensor:
    """Per-teacher temperature T_r minimizing NLL on the synthetic transfer set.

    support_only restricts the fit to samples whose label the teacher owns
    (has a prototype for) — under label skew a specialist can never be right
    outside its support, and fitting there just pushes T_r -> inf.
    Returns [R]; divide teacher r's logits by T_r to calibrate.
    """
    logits, labels, avail = (cache["teacher_logits"].float(), cache["labels"],
                             cache["available"].bool())
    temps = []
    for r in range(logits.shape[1]):
        m = avail[:, r] if support_only else torch.ones_like(labels, dtype=torch.bool)
        if m.sum() == 0:
            temps.append(1.0)
            continue
        lg, y = logits[m][:, r, :], labels[m]
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        temps.append(log_t.exp().item())
    return torch.tensor(temps)


def calibrate_cache(cache: dict, support_only: bool = False) -> dict:
    """Copy of the cache with each teacher's logits divided by its fitted T_r."""
    temps = fit_teacher_temperatures(cache, support_only)
    out = dict(cache)
    out["teacher_logits"] = cache["teacher_logits"].float() / temps.view(1, -1, 1)
    out["teacher_temperatures"] = temps
    print("fitted teacher temperatures:",
          [f"{t:.3f}" for t in temps.tolist()], flush=True)
    return out


def aggregate_cache(cache: dict, mode: str, kd_temperature: float) -> dict:
    """Robust-aggregation controls (median / trimmed mean): replace the teacher
    ensemble with a single virtual teacher whose softened distribution equals
    the coordinate-wise robust aggregate, so the standard weighted-KD machinery
    applies unchanged with uniform weights over one teacher."""
    p = torch.softmax(cache["teacher_logits"].float() / kd_temperature, dim=-1)
    if mode == "median":
        agg = p.median(dim=1).values
    elif mode == "trimmed":  # drop the min and max teacher per (sample, class)
        s, _ = p.sort(dim=1)
        agg = s[:, 1:-1, :].mean(1)
    else:
        raise ValueError(mode)
    agg = agg / agg.sum(-1, keepdim=True)
    out = dict(cache)
    out["teacher_logits"] = (kd_temperature * (agg + EPS).log()).unsqueeze(1)
    out["sims"] = torch.zeros(agg.shape[0], 1)
    out["available"] = torch.ones(agg.shape[0], 1)
    return out


def train_gate(cache: dict, epochs: int = 30, hidden: int = 256,
               lr: float = 1e-3) -> torch.Tensor:
    """IntactOFL-style learned gate: an MLP on the cached CLIP image embedding
    outputs per-sample teacher weights, trained on D_syn to minimise the NLL of
    the weighted mixture of (frozen) teacher distributions against the
    generated label. Returns weights [N, R]. The only supervision a one-shot
    server has is the generated label, so this is the learned analogue of the
    labelce control; it is in-protocol (no extra communication)."""
    x = cache["image_embeds"].float()
    x = x / x.norm(dim=-1, keepdim=True)
    p = torch.softmax(cache["teacher_logits"].float(), dim=-1)  # [N, R, C]
    y = cache["labels"]
    gate = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(),
                               torch.nn.Linear(hidden, p.shape[1]))
    opt = torch.optim.Adam(gate.parameters(), lr=lr, weight_decay=1e-4)
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            w = torch.softmax(gate(x[idx]), dim=-1)  # [B, R]
            mix = (w.unsqueeze(-1) * p[idx]).sum(1)  # [B, C]
            loss = torch.nn.functional.nll_loss((mix + EPS).log(), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return torch.softmax(gate(x), dim=-1)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return (a @ b / denom).item() if denom > 0 else float("nan")


def compute_all_weights(cache: dict, method: str, tau: float = 0.5, beta: float = 0.5,
                        missing_policy: str = "mask", blend_lambda: float = 0.5,
                        temperature_for_unc: float = 1.0,
                        client_sizes: list | None = None,
                        shuffled: bool = False) -> tuple[torch.Tensor, dict]:
    """Weights [N, R] for the whole synthetic set + reliability diagnostics (§15).

    missing_policy (prototype/joint only) — how to treat teachers with no
    prototype for label(x):
      mask      zero weight (collapses to the single owner under disjoint skew)
      ignore    raw similarity 0 participates in the softmax (soft down-weight)
      fallback  substitute the teacher's standardized uncertainty score (§7)
      blend     blend_lambda * masked weights + (1-lambda) * uniform — keeps
                the specialist-ensemble dark knowledge that pure masking loses
    """
    logits = cache["teacher_logits"].float()
    probs = torch.softmax(logits / temperature_for_unc, dim=-1)
    sims, avail, labels = cache["sims"], cache["available"], cache["labels"]

    if method == "sizeprop":
        # reviewer control: weight by client data size, ignoring all signals
        sizes = torch.tensor(client_sizes, dtype=torch.float)
        weights = (sizes / sizes.sum()).expand(logits.shape[0], -1).clone()
    elif method == "scalar-entropy":
        # reviewer control: teacher-level scalar weight (no per-sample variation)
        r_mean = normalized_entropy_reliability(probs).mean(0)
        weights = torch.softmax(r_mean / tau, dim=0).expand(
            logits.shape[0], -1).clone()
    elif method == "agreement":
        # robust-aggregation control: weight by agreement with the ensemble vote
        preds = logits.argmax(-1)
        agree = (preds.unsqueeze(2) == preds.unsqueeze(1)).float().mean(2)
        weights = torch.softmax(agree / tau, dim=1)
    elif method == "gate":
        # IntactOFL-style learned gate (see train_gate)
        weights = train_gate(cache)
    elif method == "labelce":
        # CA-MKD-style: weight by teacher log-likelihood of the generated label
        logp = torch.log_softmax(logits, dim=-1)
        s = logp.gather(2, labels.view(-1, 1, 1).expand(
            -1, logits.shape[1], 1)).squeeze(-1)
        weights = torch.softmax(s / tau, dim=1)
    elif method == "oracle":
        weights = compute_weights("uniform", probs, mask=avail)
    elif method in ("prototype", "joint") and missing_policy == "blend":
        base, _ = compute_all_weights(cache, method, tau, beta, "mask",
                                      temperature_for_unc=temperature_for_unc)
        weights = blend_lambda * base + (1.0 - blend_lambda) / base.shape[1]
    elif method in ("prototype", "joint") and missing_policy in ("ignore", "fallback"):
        runc_std = _standardize(normalized_entropy_reliability(probs), None)
        if missing_policy == "ignore":
            sem_eff = _standardize(sims, None)  # missing sims (=0) participate
        else:
            sem_eff = torch.where(avail.bool(), _standardize(sims, avail), runc_std)
        scores = sem_eff if method == "prototype" else beta * runc_std + (1 - beta) * sem_eff
        weights = torch.softmax(scores / tau, dim=1)
    else:
        mask = avail if method in ("prototype", "joint") else None
        kwargs = {"beta": beta} if method == "joint" else {}
        weights = compute_weights(method, probs, prototype_sims=sims, mask=mask,
                                  tau=tau, **kwargs)

    if shuffled:
        # control: destroys the sample-weight pairing while preserving the
        # marginal weight distribution (softening artifact vs. information)
        perm = torch.randperm(weights.shape[0],
                              generator=torch.Generator().manual_seed(1234))
        weights = weights[perm]

    correct = (logits.argmax(-1) == labels.unsqueeze(1)).float()
    w_entropy = -(weights * (weights + EPS).log()).sum(1)
    diagnostics = {
        "confidence": probs.max(-1).values, "entropy_reliability": normalized_entropy_reliability(probs),
        "sims": sims, "available": avail, "weights": weights,
        "teacher_correct": correct, "weight_entropy": w_entropy,
        "summary": {
            "method": method, "tau": tau, "beta": beta,
            "missing_policy": missing_policy, "blend_lambda": blend_lambda,
            "corr_weight_correct": _pearson(weights, correct),
            "corr_weight_sim": _pearson(weights[avail.bool().any(1)], sims[avail.bool().any(1)]),
            "mean_weight_entropy": w_entropy.mean().item(),
            "max_weight_entropy": math.log(weights.shape[1]),
            "missing_prototype_rate": 1.0 - avail.float().mean().item(),
            "mean_teacher_acc": correct.mean().item(),
        },
    }
    return weights, diagnostics


class _IndexedDataset(ManifestDataset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, y, idx


def train_global(syn_samples: list[dict], cache: dict, weights: torch.Tensor,
                 num_classes: int, backbone: str = "resnet18", epochs: int = 30,
                 lr: float = 1e-3, alpha: float = 0.5, kd_temperature: float = 3.0,
                 batch_size: int = 64, device: str | None = None,
                 pretrained: bool = True, num_workers: int = 2,
                 cosine: bool = False) -> torch.nn.Module:
    """Train the student with fixed per-sample teacher weights.

    Uses the deterministic TEST_TF transform so student inputs are pixel-
    identical to the ones the cached teacher logits were computed on.
    """
    import os
    num_workers = int(os.environ.get("UA_WORKERS", num_workers))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(backbone, num_classes, pretrained).to(device)
    if os.environ.get("UA_PRELOAD") == "1":
        # The distillation input transform is deterministic, so decode D_syn
        # once per work dir and reuse the tensor cache (AV-scanning and PNG
        # decode otherwise starve the GPU on local Windows runs).
        from torch.utils.data import TensorDataset
        pp = os.path.join(os.environ["UA_WORK"], "preload.pt")
        x = None
        if os.path.exists(pp):
            x = torch.load(pp, map_location="cpu", weights_only=False)
            if len(x) != len(syn_samples):
                x = None
        if x is None:
            dec = DataLoader(ManifestDataset(syn_samples, TEST_TF),
                             batch_size=256, num_workers=num_workers)
            x = torch.cat([b for b, _ in tqdm(dec, desc="preload")])
            torch.save(x, pp)
        y = torch.tensor([s["label"] for s in syn_samples])
        ds = TensorDataset(x, y, torch.arange(len(y)))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)
    else:
        loader = DataLoader(_IndexedDataset(syn_samples, TEST_TF),
                            batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    t_logits_all = cache["teacher_logits"]
    model.train()
    for epoch in range(epochs):
        loss_sum, n = 0.0, 0
        for x, y, idx in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = weighted_kd_loss(model(x), t_logits_all[idx].to(device), y,
                                    weights[idx].to(device), alpha=alpha,
                                    temperature=kd_temperature)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
        if sched:
            sched.step()
        print(f"epoch {epoch}: loss {loss_sum / n:.4f}")
    return model
