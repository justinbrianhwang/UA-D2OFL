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

## Settings

All settings use NICO++ (Common) with 60 classes and 6 clients,
ImageNet-pretrained ResNet-18 teachers/student.

| Setting | Description |
|---|---|
| `f6r` | Symmetric feature skew (client = domain, equal 100 imgs/class) |
| `a60m` | Mild data-size asymmetry — per-client caps [100, 70, 50, 35, 25, 18] imgs/class |
| `a60n` | Label-noise asymmetry — per-client flip rates [0, .1, .2, .3, .4, .5] |
| `a60` | Strong data-size asymmetry — per-client caps [100, 50, 25, 12, 6, 3] imgs/class |
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

Data released under CC BY 4.0. Cite the paper above if you use it.
