"""Paper-scale phase 1 data: NICO++ Common, ALL 60 classes, 6 domains.

Produces one image-pool zip + two manifest files:
  feature-skew manifests: client = domain, all 60 classes
  label-skew manifests:   6 clients x 10 disjoint classes, all domains (paper setting)
Train cap 100/cell (~52k images); test = held-out 20% per cell.
"""

import json
import os
import random
import zipfile
from collections import defaultdict

OUTER = r"G:/내 드라이브/Dataset/NICOpp/NICOpp_dropbox.zip"
INNER = "DG_Benchmark/NICO_DG_Benchmark.zip"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
SEED = 0
TRAIN_CAP = 100

outer = zipfile.ZipFile(OUTER)
z = zipfile.ZipFile(outer.open(outer.getinfo(INNER)))
cells = defaultdict(list)
for n in z.namelist():
    parts = n.split("/")
    if len(parts) == 4 and parts[3].lower().endswith((".jpg", ".jpeg", ".png")):
        cells[(parts[1], parts[2])].append(n)
domains = sorted({d for d, _ in cells})
classes = sorted(set.intersection(*[{c for d2, c in cells if d2 == d} for d in domains]))
print(f"{len(classes)} classes, {len(domains)} domains")
class_idx = {c: i for i, c in enumerate(classes)}

split_cache = {}
for (domain, c), names in cells.items():
    if c not in class_idx:
        continue
    names = sorted(names)
    random.Random(f"{SEED}/{domain}/{c}").shuffle(names)
    cut = int(len(names) * 0.8)
    split_cache[(domain, c)] = (names[:cut][:TRAIN_CAP], names[cut:])

# feature-skew: client = domain index
ftrain, ftest = defaultdict(list), defaultdict(list)
for ci, domain in enumerate(domains):
    for c in classes:
        tr, te = split_cache[(domain, c)]
        for names, split in [(tr, ftrain), (te, ftest)]:
            for n in names:
                split[ci].append({"path": n, "label": class_idx[c], "domain": domain})
json.dump({"classes": classes, "domains": domains, "mode": "feature60",
           "train": {str(k): v for k, v in ftrain.items()},
           "test": {str(k): v for k, v in ftest.items()}},
          open(os.path.join(SCRATCH, "manifest_f60.json"), "w"))
print("feature-skew:", {k: (len(ftrain[k]), len(ftest[k])) for k in ftrain})

# label-skew: 10 disjoint classes per client, all domains (paper setting)
owner = {c: class_idx[c] // 10 for c in classes}
ltrain, ltest = defaultdict(list), defaultdict(list)
for c in classes:
    r = owner[c]
    for domain in domains:
        tr, te = split_cache[(domain, c)]
        for names, split in [(tr, ltrain), (te, ltest)]:
            for n in names:
                split[r].append({"path": n, "label": class_idx[c], "domain": domain})
json.dump({"classes": classes, "domains": domains, "mode": "label60",
           "train": {str(k): v for k, v in ltrain.items()},
           "test": {str(k): v for k, v in ltest.items()}},
          open(os.path.join(SCRATCH, "manifest_l60.json"), "w"))
print("label-skew:", {k: (len(ltrain[k]), len(ltest[k])) for k in ltrain})

keep = sorted({s["path"] for split in [ftrain, ftest] for v in split.values() for s in v})
out_path = os.path.join(SCRATCH, "nico60_pool.zip")
with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as out:
    for i, n in enumerate(keep):
        out.writestr(n, z.read(n))
        if (i + 1) % 5000 == 0:
            print(f"{i + 1}/{len(keep)}", flush=True)
print(f"done: {out_path} {os.path.getsize(out_path) / 1e9:.2f} GB, {len(keep)} images")
