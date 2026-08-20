"""Accuracy + calibration metrics: ECE, NLL, Brier."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

EPS = 1e-12


def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected calibration error over max-prob confidence bins."""
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.float().mean().item() * abs(
                correct[in_bin].mean().item() - conf[in_bin].mean().item())
    return total


def nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return -(probs[torch.arange(len(labels)), labels] + EPS).log().mean().item()


def brier(probs: torch.Tensor, labels: torch.Tensor) -> float:
    onehot = F.one_hot(labels, probs.shape[1]).float()
    return ((probs - onehot) ** 2).sum(dim=1).mean().item()


@torch.no_grad()
def evaluate(model, dataset, batch_size: int = 256, device: str | None = None) -> dict:
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    all_probs, all_labels = [], []
    for x, y in loader:
        all_probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        all_labels.append(y)
    probs, labels = torch.cat(all_probs), torch.cat(all_labels)
    return {"acc": (probs.argmax(1) == labels).float().mean().item(),
            "ece": ece(probs, labels), "nll": nll(probs, labels),
            "brier": brier(probs, labels), "n": len(labels)}
