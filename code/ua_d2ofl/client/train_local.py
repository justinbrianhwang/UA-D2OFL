"""Local teacher training (D2OFL Eq. 17): ResNet-18, Adam lr=1e-3, CE.

Usage: python -m ua_d2ofl.client.train_local --manifest client0.json \
    --num_classes 60 --out teacher0.pt
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ua_d2ofl.data.dataset import ManifestDataset, TRAIN_TF, load_manifest
from ua_d2ofl.models import initialize_model


def train_teacher(samples: list[dict], num_classes: int, backbone: str = "resnet18",
                  epochs: int = 30, lr: float = 1e-3, batch_size: int = 64,
                  device: str | None = None, num_workers: int = 4,
                  pretrained: bool = True, weight_decay: float = 0.0,
                  cosine: bool = False) -> nn.Module:
    import os
    num_workers = int(os.environ.get("UA_WORKERS", num_workers))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(backbone, num_classes, pretrained).to(device)
    loader = DataLoader(ManifestDataset(samples, TRAIN_TF), batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        if sched:
            sched.step()
        print(f"epoch {epoch}: loss {loss_sum/total:.4f} acc {correct/total:.4f}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    model = train_teacher(load_manifest(a.manifest), a.num_classes, a.backbone,
                          a.epochs, a.lr, a.batch_size)
    torch.save({"backbone": a.backbone, "num_classes": a.num_classes,
                "state_dict": model.state_dict()}, a.out)
    print(f"saved -> {a.out}")
