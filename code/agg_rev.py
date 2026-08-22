"""Revision aggregation: l6s label-skew (strong recipe), a60 controls +
5-seed stats, a60m sizeprop. Paired t-tests and sign tests vs uniform."""
import json
import os
import statistics as st

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts_paper")


def load(tag):
    res = json.load(open(os.path.join(ART, tag, "results_paper.json")))
    by = {}
    for k, v in res.items():
        *cfg, s = k.split("|")
        by.setdefault("|".join(cfg), {})[s] = v
    return by


def row(by, cfg, uni="uniform"):
    seeds = by[cfg]
    accs = [v["overall"]["acc"] for v in seeds.values()]
    eces = [v["overall"]["ece"] for v in seeds.values()]
    nlls = [v["overall"].get("nll") for v in seeds.values()]
    briers = [v["overall"].get("brier") for v in seeds.values()]
    out = (f"{cfg:<22} acc {st.mean(accs):.4f}±{st.stdev(accs):.4f} "
           f"ece {st.mean(eces):.4f} nll {st.mean(nlls):.3f} "
           f"brier {st.mean(briers):.4f} n={len(accs)}")
    if cfg != uni and uni in by:
        pairs = [(seeds[s]["overall"]["acc"], by[uni][s]["overall"]["acc"])
                 for s in seeds if s in by[uni]]
        if len(pairs) >= 3:
            d = [a - b for a, b in pairs]
            t, p = stats.ttest_rel([a for a, _ in pairs], [b for _, b in pairs])
            pos = sum(1 for x in d if x > 0)
            sign_p = stats.binomtest(pos, len(d), 0.5).pvalue
            out += (f"  D={100*st.mean(d):+.2f}pp t={t:.2f} p={p:.3f} "
                    f"sign {pos}/{len(d)} p={sign_p:.3f}")
    return out


print("===== l6s: label-skew, strong recipe (F2 fix) =====")
by = load("l6r")
for cfg in ["uniform", "joint|blend0.25", "confidence", "entropy",
            "prototype|blend0.5", "joint|mask", "oracle"]:
    print(row(by, cfg))

print("\n===== a60: strong size asymmetry — controls + 5 seeds =====")
by = load("a60")
for cfg in ["uniform", "entropy", "sizeprop", "scalar-entropy", "labelce",
            "entropy|shuffled", "confidence", "joint|mask", "uniform+studentTS"]:
    if cfg in by:
        print(row(by, cfg))
# student TS detail
if "uniform+studentTS" in by:
    for s, v in sorted(by["uniform+studentTS"].items()):
        print(f"  studentTS {s}: raw ece {v['raw']['ece']:.4f} -> "
              f"ts ece {v['overall']['ece']:.4f} (T={v['fitted_T']:.3f})")

print("\n===== a60m: mild asymmetry — sizeprop control =====")
by = load("a60m")
for cfg in ["uniform", "entropy", "sizeprop"]:
    if cfg in by:
        print(row(by, cfg))
