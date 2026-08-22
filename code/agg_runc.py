import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
"""Run C aggregation: teacher-realization pooling (t0/t1/t2), robust
aggregation controls (a60, a60n), lambda sweep (l6r), Holm correction."""
import statistics as st
from scipy import stats
import agg_rev; load, row = agg_rev.load, agg_rev.row

def acc(by, cfg, s): return by[cfg][s]["overall"]["acc"]

def paired(d, label):
    t, p = stats.ttest_rel(d, [0]*len(d)) if len(d) > 1 else (float("nan"), 1)
    pos = sum(x > 0 for x in d)
    sp = stats.binomtest(pos, len(d), 0.5).pvalue
    print(f"{label:<28} D={100*st.mean(d):+.2f}pp sd={100*st.stdev(d):.2f} "
          f"t={t:.2f} p={p:.3f} sign {pos}/{len(d)} p={sp:.3f} n={len(d)}")
    return p

print("===== full-pipeline variance: teacher realizations t0/t1/t2 (a60) =====")
reals = {"t0": load("a60"), "t1": load("a60t1"), "t2": load("a60t2")}
pvals = {}
for cfg in ["entropy", "sizeprop"]:
    d = []
    for t, by in reals.items():
        for s in ("s0", "s1"):
            dd = acc(by, cfg, s) - acc(by, "uniform", s)
            d.append(dd); print(f"  {cfg:<9} {t} {s}: {100*dd:+.2f}pp  (uni {acc(by,'uniform',s):.4f})")
    pvals[f"{cfg} pooled 3x2"] = paired(d, f"{cfg} vs uniform pooled")
    # per-realization means
    for t, by in reals.items():
        m = st.mean(acc(by, cfg, s) - acc(by, "uniform", s) for s in ("s0", "s1"))
        print(f"    {t} mean {100*m:+.2f}pp")
# t0 5-seed (original)
for cfg in ["entropy", "sizeprop", "scalar-entropy"]:
    by = reals["t0"]
    d = [acc(by, cfg, s) - acc(by, "uniform", s) for s in sorted(by[cfg]) if s in by["uniform"]]
    pvals[f"{cfg} t0 5seed"] = paired(d, f"{cfg} vs uniform t0 5-seed")
# realization spread of uniform itself
u = [acc(by, "uniform", s) for by in reals.values() for s in ("s0", "s1")]
print(f"uniform across realizations: {st.mean(u):.4f} sd {st.stdev(u):.4f} "
      f"(t means: {[round(st.mean(acc(b,'uniform',s) for s in ('s0','s1')),4) for b in reals.values()]})")

print("\n===== robust aggregation controls =====")
for tag in ["a60", "a60n"]:
    print(f"--- {tag}")
    by = load(tag)
    for cfg in ["uniform", "entropy", "median", "trimmed", "agreement", "joint|mask"]:
        if cfg in by: print(row(by, cfg))

print("\n===== lambda sweep (l6r label skew) =====")
by = load("l6r")
for cfg in ["uniform", "joint|blend0.1", "joint|blend0.25", "joint|blend0.5",
            "joint|blend0.75", "joint|mask", "oracle"]:
    print(row(by, cfg))

print("\n===== Holm over headline comparisons =====")
names = list(pvals); ps = [pvals[n] for n in names]
order = sorted(range(len(ps)), key=lambda i: ps[i]); m = len(ps); adj = [0]*m; run = 0
for k, i in enumerate(order):
    run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
for n, p, a in zip(names, ps, adj):
    print(f"{n:<24} p={p:.3f} holm={a:.3f} {'*' if a < .05 else ''}")

print("\n===== realization-level (n=3 realization means; primary test in the paper) =====")
pr = {}
for cfg in ["entropy", "sizeprop"]:
    per = [st.mean(acc(by, cfg, s) - acc(by, "uniform", s) for s in ("s0", "s1")) for by in reals.values()]
    t, p = stats.ttest_1samp(per, 0); se = st.stdev(per) / 3 ** 0.5; h = stats.t.ppf(.975, 2) * se
    print(f"{cfg:<9} acc  mean {100*st.mean(per):+.2f}pp CI [{100*(st.mean(per)-h):+.2f},{100*(st.mean(per)+h):+.2f}] t={t:.2f} p={p:.3f}")
    pr[cfg + " acc"] = p
per = [st.mean(by["entropy"][s]["overall"]["ece"] - by["uniform"][s]["overall"]["ece"] for s in ("s0", "s1")) for by in reals.values()]
t, p = stats.ttest_1samp(per, 0); print(f"entropy   dECE mean {st.mean(per):+.4f} p={p:.3f}"); pr["entropy ece"] = p
run = 0
for k, n in enumerate(sorted(pr, key=pr.get)):
    run = max(run, min(1, (3 - k) * pr[n])); print(f"  Holm(3) {n:<12} p={pr[n]:.3f} -> {run:.3f}")

print("\n===== power of the pooled (n=6) entropy test to detect +0.51pp =====")
from scipy.stats import nct
pooled = [acc(by, "entropy", s) - acc(by, "uniform", s) for by in reals.values() for s in ("s0", "s1")]
sd = st.stdev(pooled); ncp = 0.0051 / (sd / 6 ** 0.5); crit = stats.t.ppf(.975, 5)
print(f"sd={100*sd:.2f}pp  power={1 - nct.cdf(crit, 5, ncp) + nct.cdf(-crit, 5, ncp):.2f}")
