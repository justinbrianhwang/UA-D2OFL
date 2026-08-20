"""Dose-response variants of the asymmetric-client setup (B-2 extension).

  a60m  mild size asymmetry: caps [100, 70, 50, 35, 25, 18] (~5.6x spread;
        a60 was [100, 50, 25, 12, 6, 3] = 33x)
  a60n  equal sizes, label-noise asymmetry: client r gets rate
        [0, .1, .2, .3, .4, .5] of train labels flipped to a wrong class

Test sets stay clean/unchanged. Prototypes remain f60's in both variants.
"""
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0


def subsample(m, caps):
    for r_str, samples in m["train"].items():
        r = int(r_str)
        per_class = defaultdict(list)
        for s in samples:
            per_class[s["label"]].append(s)
        kept = []
        for label in sorted(per_class):
            ss = sorted(per_class[label], key=lambda s: s["path"])
            random.Random(f"{SEED}/a60m/{r}/{label}").shuffle(ss)
            kept.extend(ss[:caps[r]])
        m["train"][r_str] = kept
    return m


def add_noise(m, rates):
    C = len(m["classes"])
    flipped = {}
    for r_str, samples in m["train"].items():
        r = int(r_str)
        rng = random.Random(f"{SEED}/a60n/{r}")
        n_flip = 0
        for s in samples:
            if rng.random() < rates[r]:
                wrong = rng.randrange(C - 1)
                s["label"] = wrong if wrong < s["label"] else wrong + 1
                n_flip += 1
        flipped[r] = n_flip
    print("flipped:", flipped)
    return m


base = json.load(open(os.path.join(HERE, "manifest_f60.json")))
m = subsample(json.loads(json.dumps(base)), [100, 70, 50, 35, 25, 18])
m["mode"] = "asym60_mild"
json.dump(m, open(os.path.join(HERE, "manifest_a60m.json"), "w"))
print("a60m train:", {r: len(v) for r, v in sorted(m["train"].items())})

m = add_noise(json.loads(json.dumps(base)), [0, .1, .2, .3, .4, .5])
m["mode"] = "asym60_noise"
json.dump(m, open(os.path.join(HERE, "manifest_a60n.json"), "w"))
print("a60n train:", {r: len(v) for r, v in sorted(m["train"].items())})
