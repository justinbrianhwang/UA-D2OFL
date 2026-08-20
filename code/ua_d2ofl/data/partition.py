"""Client partitioning for non-IID experiments.

A manifest is a list of samples: dicts with keys {path, label, domain}
(label = int class index, domain = str). Builders return
{client_id: [sample, ...]}. All partitions are deterministic given seed.
"""

import os
import random
from collections import defaultdict

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def build_manifest(root: str, classes: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """Scan a root/domain/category/*.jpg tree (NICO++/DomainNet layout)."""
    domains = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if classes is None:
        classes = sorted({c for d in domains for c in os.listdir(os.path.join(root, d))
                          if os.path.isdir(os.path.join(root, d, c))})
    class_idx = {c: i for i, c in enumerate(classes)}
    manifest = []
    for d in domains:
        for c in classes:
            cdir = os.path.join(root, d, c)
            if not os.path.isdir(cdir):
                continue
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith(IMG_EXTS):
                    manifest.append({"path": os.path.join(cdir, f),
                                     "label": class_idx[c], "domain": d})
    return manifest, classes


def feature_skew(manifest: list[dict]) -> dict[int, list[dict]]:
    """One domain per client (D2OFL feature-based non-IID)."""
    by_domain = defaultdict(list)
    for s in manifest:
        by_domain[s["domain"]].append(s)
    return {i: by_domain[d] for i, d in enumerate(sorted(by_domain))}


def label_skew(manifest: list[dict], num_clients: int,
               classes_per_client: int | None = None, seed: int = 0) -> dict[int, list[dict]]:
    """Disjoint class blocks per client, all domains included (D2OFL label-based non-IID)."""
    labels = sorted({s["label"] for s in manifest})
    rng = random.Random(seed)
    rng.shuffle(labels)
    if classes_per_client is None:
        classes_per_client = len(labels) // num_clients
    assign = {}
    for i in range(num_clients):
        for lb in labels[i * classes_per_client:(i + 1) * classes_per_client]:
            assign[lb] = i
    clients = {i: [] for i in range(num_clients)}
    for s in manifest:
        if s["label"] in assign:
            clients[assign[s["label"]]].append(s)
    return clients


def dirichlet_skew(manifest: list[dict], num_clients: int, alpha: float,
                   seed: int = 0) -> dict[int, list[dict]]:
    """Dirichlet(alpha) label partition for controlled heterogeneity severity."""
    import numpy as np
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for s in manifest:
        by_label[s["label"]].append(s)
    clients = {i: [] for i in range(num_clients)}
    for label, samples in sorted(by_label.items()):
        samples = sorted(samples, key=lambda s: s["path"])
        rng.shuffle(samples)
        props = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(props) * len(samples)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(np.array(samples, dtype=object), cuts)):
            clients[i].extend(chunk.tolist())
    return clients
