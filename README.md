# UA-D²OFL — Experimental Results

Full experimental results for the paper:

> **When Does Reliability-Weighted Multi-Teacher Distillation Help
> Diffusion-Assisted One-Shot Federated Learning?**
> Sunjun Hwang, Independent Research. (In submission.)

The study extends D²OFL-style one-shot federated learning (BLIP-2 captioning →
CLIP prototypes → Stable Diffusion synthetic transfer set → multi-teacher
distillation) with sample-wise teacher-reliability weighting, and maps the
conditions under which adaptive weighting beats the uniform ensemble average.

## Files

| File | Contents |
|---|---|
| `results/all_results.csv` | Every (setting, config, seed) run: top-1 accuracy, ECE, NLL, Brier, mean weight entropy, corr(weight, teacher-correctness), missing-prototype rate |
| `results/teacher_quality.csv` | Per-client teacher accuracy (own-domain test / pooled global test) for each asymmetry setting |
| `results/<setting>.json` | Raw per-setting result files, including per-client evaluation breakdowns |
| `code/ua_d2ofl/` | Full pipeline implementation (Python / PyTorch, MIT license) |
| `code/agg_rev.py`, `code/agg_runc.py` | Aggregation scripts reproducing every paired test, sign test, and Holm correction in the paper from `results/<setting>.json` |
| `code/paper_recipe.env` | **Exact env-var training recipe used for every paper result** (module defaults are weaker legacy values) |
| `code/data_prep/` | Dataset pool / manifest builders (NICO++ paths are environment-specific) |
| `code/requirements.txt` | Python dependencies (PyTorch preinstalled separately) |
| `manifests/` | The exact NICO++ train/test split manifests for every setting (paths relative to the extracted NICO_DG pool) |

## Code layout

```
code/ua_d2ofl/
  client/        BLIP-2 captioning, CLIP prototype encoding (+ Mahalanobis
                 filtering), local teacher training
  server/        Stable Diffusion generation, signal cache (precompute),
                 reliability weighting + weighted distillation (distill.py)
  distillation/  reliability estimators and the weighted KD loss
  data/          manifest datasets, partition builders
  metrics/       accuracy / ECE / NLL / Brier
  experiments/   stage runners (smoke.py), paper runs (paper_run.py),
                 student-side temperature-scaling control (student_ts.py)
  tests/         unit tests (python -m ua_d2ofl.tests.test_reliability)
```

Stages run as `python -m ua_d2ofl.experiments.smoke --stage
data,captions,encode,teachers,generate,cache` followed by
`python -m ua_d2ofl.experiments.paper_run --seed S --config C`; work
directory, synthetic budget, and training recipe are set via `UA_*`
environment variables (see the module headers).

## Settings

All settings use NICO++ (Common) with 60 classes and 6 clients,
ImageNet-pretrained ResNet-18 teachers/student.

| Setting | Description |
|---|---|
| `f6r` | Symmetric feature skew (client = domain, equal 100 imgs/class) |
| `a60m` | Mild data-size asymmetry — per-client caps [100, 70, 50, 35, 25, 18] imgs/class |
| `a60n` | Label-noise asymmetry — per-client flip rates [0, .1, .2, .3, .4, .5] |
| `a60` | Strong data-size asymmetry — per-client caps [100, 50, 25, 12, 6, 3] imgs/class (teacher realization t0) |
| `a60t1`, `a60t2` | Same as `a60` with teachers re-trained under teacher seeds 1 and 2 (`UA_TEACHER_SEED`), same D_syn — full-pipeline variance check |
| `l60_180` | Disjoint label skew (10 classes/client), 180 synthetic imgs/prototype |
| `l6r` | Label skew under the strong training recipe, incl. per-teacher temperature-calibration variants |
| `f60`, `l60` | Legacy runs under the original (weaker) training recipe, kept for completeness |

## Configs

`uniform` (D²OFL-style ensemble average), `confidence`, `entropy`, `prototype`,
`joint` (β-mix of uncertainty and CLIP-prototype relevance) — with
missing-prototype policies `mask` (hard) and `blendλ` (soft mixture with
uniform); `oracle` = uniform over the teachers that own the class (class-mask
baseline); `cal`/`calsup` = per-teacher temperature calibration fitted on the
synthetic transfer set (all-samples / in-support).

## License

Data (`results/`, `manifests/`) released under CC BY 4.0; code (`code/`)
under MIT (see `code/LICENSE`). Cite the paper above if you use either.
