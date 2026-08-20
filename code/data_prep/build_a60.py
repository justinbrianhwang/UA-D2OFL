"""Asymmetric-client manifest (B-2): f60 feature-skew with per-client train
caps [100, 50, 25, 12, 6, 3] imgs/class — geometric data-size asymmetry so
teacher quality genuinely differs. Test sets unchanged (full f60 test).
Prototypes/D_syn stay f60's, isolating the teacher-quality axis.
"""
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CAPS = [100, 50, 25, 12, 6, 3]
SEED = 0

m = json.load(open(os.path.join(HERE, "manifest_f60.json")))
for r_str, samples in m["train"].items():
    r = int(r_str)
    per_class = defaultdict(list)
    for s in samples:
        per_class[s["label"]].append(s)
    kept = []
    for label in sorted(per_class):
        ss = sorted(per_class[label], key=lambda s: s["path"])
        random.Random(f"{SEED}/a60/{r}/{label}").shuffle(ss)
        kept.extend(ss[:CAPS[r]])
    m["train"][r_str] = kept
m["mode"] = "asym60"
json.dump(m, open(os.path.join(HERE, "manifest_a60.json"), "w"))
print("train sizes:", {r: len(v) for r, v in sorted(m["train"].items())})
print("test sizes :", {r: len(v) for r, v in sorted(m["test"].items())})
