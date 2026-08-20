"""End-to-end smoke run on a NICO++ Common subset. Runs ON the Colab VM.

10 classes x 6 domains (feature-skew, client = domain). Every stage is
idempotent and writes artifacts to a Drive work dir, so a killed session
resumes where it left off. Stages:
  data      extract subset from the Dropbox zip on Drive -> VM local disk
  captions  BLIP2 captions per client
  encode    CLIP prototypes (cond + pooled) with Mahalanobis filtering
  teachers  local ResNet-18 per client
  generate  Stable Diffusion synthetic set
  cache     teacher logits / CLIP image embeds / prototype sims
  distill   train + evaluate all weighting methods
  report    print result table from results.json

Usage (on VM): python -m ua_d2ofl.experiments.smoke --stage all
"""

import argparse
import json
import os
import random
import zipfile
from collections import defaultdict

import torch

DRIVE = "/content/drive/MyDrive"
NICO_ZIP = os.environ.get("UA_NICO_ZIP", f"{DRIVE}/Dataset/NICOpp/NICOpp_dropbox.zip")
SUBSET_ZIP = os.environ.get("UA_SUBSET_ZIP", "/content/nico_subset.zip")
INNER = "DG_Benchmark/NICO_DG_Benchmark.zip"
WORK = os.environ.get("UA_WORK", f"{DRIVE}/UA_D2OFL/smoke_nico")
DATA = os.environ.get("UA_DATA", "/content/nico")

NUM_CLASSES = 10      # feature-mode subset build only; runtime stages use len(manifests classes)
SEED = 0
CAPTION_CAP = int(os.environ.get("UA_CAPTION_CAP", "30"))  # captions per (client, class)
TRAIN_CAP = 100       # teacher-training images per (client, class)
TEACHER_EPOCHS = int(os.environ.get("UA_TEACHER_EPOCHS", "20"))
SYN_PER_PROTO = int(os.environ.get("UA_SYN_PER_PROTO", "10"))
SD_STEPS = 30
DISTILL_EPOCHS = 15
METHODS = ["uniform", "oracle", "confidence", "entropy", "prototype", "joint"]
TAU, BETA, ALPHA, KD_T = 0.5, 0.5, 0.5, 3.0


def _manifests():
    with open(f"{WORK}/manifests.json", encoding="utf-8") as f:
        m = json.load(f)
    for split in ["train", "test"]:  # stored paths are DATA-relative
        for samples in m[split].values():
            for s in samples:
                if not os.path.isabs(s["path"]):
                    s["path"] = os.path.join(DATA, s["path"])
    return m


def stage_data():
    """Get the 10-class subset. Preferred: a prebuilt subset zip (uploaded via
    colab upload) containing images + manifests.json with DATA-relative paths.
    Fallback: extract from the nested (STORED) Dropbox zip on mounted Drive."""
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(SUBSET_ZIP):
        with zipfile.ZipFile(SUBSET_ZIP) as z:
            z.extractall(DATA)
        os.replace(os.path.join(DATA, "manifests.json"), f"{WORK}/manifests.json")
        m = _manifests()
        print("subset extracted:",
              {k: (len(m["train"][k]), len(m["test"][k])) for k in m["train"]}, flush=True)
        return
    local_inner = "/content/NICO_DG_Benchmark.zip"
    if not os.path.exists(local_inner):
        print("copying inner zip from Drive ...", flush=True)
        with zipfile.ZipFile(NICO_ZIP) as outer, outer.open(INNER) as src, \
                open(local_inner, "wb") as dst:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
    z = zipfile.ZipFile(local_inner)
    cells = defaultdict(list)  # (domain, class) -> [names]
    for n in z.namelist():
        parts = n.split("/")
        if len(parts) == 4 and parts[3].lower().endswith((".jpg", ".jpeg", ".png")):
            cells[(parts[1], parts[2])].append(n)
    domains = sorted({d for d, _ in cells})
    common = sorted(set.intersection(*[{c for d2, c in cells if d2 == d} for d in domains]))
    classes = common[:NUM_CLASSES]
    print(f"domains={domains}\nclasses={classes}", flush=True)

    train, test = defaultdict(list), defaultdict(list)  # client -> samples
    class_idx = {c: i for i, c in enumerate(classes)}
    for ci, domain in enumerate(domains):
        for c in classes:
            names = sorted(cells[(domain, c)])
            random.Random(f"{SEED}/{domain}/{c}").shuffle(names)
            cut = int(len(names) * 0.8)
            for split, chosen in [(train, names[:cut][:TRAIN_CAP]), (test, names[cut:])]:
                for n in chosen:
                    if not os.path.exists(os.path.join(DATA, n)):
                        z.extract(n, DATA)
                    split[ci].append({"path": n, "label": class_idx[c], "domain": domain})
    out = {"classes": classes, "domains": domains,
           "train": {str(k): v for k, v in train.items()},
           "test": {str(k): v for k, v in test.items()}}
    with open(f"{WORK}/manifests.json", "w", encoding="utf-8") as f:
        json.dump(out, f)
    print({k: (len(train[int(k)]), len(test[int(k)])) for k in out["train"]}, flush=True)


def _client_caption_manifest(m, r):
    per_class = defaultdict(list)
    for s in m["train"][str(r)]:
        per_class[s["label"]].append(s)
    return [s for ss in per_class.values() for s in ss[:CAPTION_CAP]]


def _clients(m):
    """All clients, or just $UA_CLIENT — lets the runner chunk long stages."""
    if "UA_CLIENT" in os.environ:
        return [int(os.environ["UA_CLIENT"])]
    return list(range(len(m["domains"])))


def stage_captions():
    from ua_d2ofl.client.captioning import caption_manifest
    m = _manifests()
    for r in _clients(m):
        mpath = f"{WORK}/caption_manifest{r}.json"
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(_client_caption_manifest(m, r), f)
        caption_manifest(mpath, f"{WORK}/captions{r}.jsonl")


def stage_encode():
    from ua_d2ofl.client.encoding import build_prototypes
    m = _manifests()
    for r in range(len(m["domains"])):
        out = f"{WORK}/prototypes{r}.pt"
        if os.path.exists(out):
            continue
        torch.save(build_prototypes(f"{WORK}/captions{r}.jsonl", m["classes"]), out)
        print(f"prototypes{r}.pt saved", flush=True)


def stage_teachers():
    from ua_d2ofl.client.train_local import train_teacher
    m = _manifests()
    for r in _clients(m):
        out = f"{WORK}/teacher{r}.pt"
        if os.path.exists(out):
            continue
        model = train_teacher(
            m["train"][str(r)], len(m["classes"]), epochs=TEACHER_EPOCHS,
            lr=float(os.environ.get("UA_TEACHER_LR", "1e-3")),
            weight_decay=float(os.environ.get("UA_TEACHER_WD", "0")),
            cosine=os.environ.get("UA_TEACHER_COS") == "1")
        torch.save({"backbone": "resnet18", "num_classes": len(m["classes"]),
                    "state_dict": model.state_dict()}, out)
        print(f"teacher{r}.pt saved", flush=True)


def stage_generate():
    from ua_d2ofl.server.generation import generate_dataset
    m = _manifests()
    files = {r: f"{WORK}/prototypes{r}.pt" for r in _clients(m)}
    generate_dataset(files, f"{WORK}/D_syn", num_images=SYN_PER_PROTO, steps=SD_STEPS)


def stage_cache():
    from ua_d2ofl.server.generation import load_syn_manifest
    from ua_d2ofl.server.precompute import build_cache
    m = _manifests()
    syn = load_syn_manifest(f"{WORK}/D_syn")
    r_count = len(m["domains"])
    expected = sum(
        len(torch.load(f"{WORK}/prototypes{r}.pt", map_location="cpu",
                       weights_only=False)) * SYN_PER_PROTO
        for r in range(r_count))
    if len(syn) != expected:
        raise RuntimeError(
            f"D_syn incomplete: {len(syn)} images, expected {expected} — "
            "regenerate before caching")
    cache = build_cache(syn, [f"{WORK}/teacher{r}.pt" for r in range(r_count)],
                        {r: f"{WORK}/prototypes{r}.pt" for r in range(r_count)})
    torch.save({"cache": cache, "syn": syn}, f"{WORK}/cache.pt")


def stage_distill():
    from ua_d2ofl.data.dataset import ManifestDataset
    from ua_d2ofl.metrics import evaluate
    from ua_d2ofl.server.distill import compute_all_weights, train_global
    m = _manifests()
    blob = torch.load(f"{WORK}/cache.pt", map_location="cpu", weights_only=False)
    cache, syn = blob["cache"], blob["syn"]
    results = {}
    if os.path.exists(f"{WORK}/results.json"):
        with open(f"{WORK}/results.json", encoding="utf-8") as f:
            results = json.load(f)
    for method in METHODS:
        if method in results:
            continue
        torch.manual_seed(SEED)
        weights, diag = compute_all_weights(cache, method, tau=TAU, beta=BETA)
        model = train_global(syn, cache, weights, len(m["classes"]), epochs=DISTILL_EPOCHS,
                             alpha=ALPHA, kd_temperature=KD_T)
        per_domain = {}
        for r, domain in enumerate(m["domains"]):
            per_domain[domain] = evaluate(model, ManifestDataset(m["test"][str(r)]))
        overall = {k: sum(d[k] * d["n"] for d in per_domain.values()) /
                      sum(d["n"] for d in per_domain.values())
                   for k in ["acc", "ece", "nll", "brier"]}
        results[method] = {"per_domain": per_domain, "overall": overall,
                           "diag": diag["summary"]}
        with open(f"{WORK}/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"{method}: acc={overall['acc']:.4f} ece={overall['ece']:.4f} "
              f"H_w={diag['summary']['mean_weight_entropy']:.3f}", flush=True)


def stage_report():
    with open(f"{WORK}/results.json", encoding="utf-8") as f:
        results = json.load(f)
    print(f"{'method':<12}{'acc':>8}{'ece':>8}{'nll':>8}{'brier':>8}"
          f"{'H_w':>8}{'corr_wc':>9}")
    for method, r in results.items():
        o, d = r["overall"], r["diag"]
        print(f"{method:<12}{o['acc']:>8.4f}{o['ece']:>8.4f}{o['nll']:>8.3f}"
              f"{o['brier']:>8.4f}{d['mean_weight_entropy']:>8.3f}"
              f"{d['corr_weight_correct']:>9.3f}")


STAGES = {"data": stage_data, "captions": stage_captions, "encode": stage_encode,
          "teachers": stage_teachers, "generate": stage_generate,
          "cache": stage_cache, "distill": stage_distill, "report": stage_report}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   help="comma-separated stage names or 'all'")
    a = p.parse_args()
    todo = list(STAGES) if a.stage == "all" else a.stage.split(",")
    for name in todo:
        print(f"===== stage: {name} =====", flush=True)
        STAGES[name]()
    print("smoke run complete")
