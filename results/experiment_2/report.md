# Experiment 2: Contrastive Retrieval SHAP for CLIP

## Question and controlled setup

Does explaining the positive-minus-hard-negative cosine margin identify image regions whose removal destroys CLIP’s positive-caption preference faster than explaining positive-caption cosine alone?

Frozen `openai/clip-vit-base-patch32`, Hugging Face Transformers. Pointwise SHAP explains `cosine(I,T+)`; CR-SHAP explains `cosine(I,T+) - cosine(I,T−)`. Both use the unchanged Experiment 1 constrained Kernel SHAP solver, 128 identical seeded coalitions, the same mean-masked images, and one image embedding per coalition for both caption scores. No LaMa, mask expansion, training, or additional method was used.

SLIC settings: `n_segments=16`, `compactness=30`, `start_label=0`, `channel_axis=-1`; all other parameters retain scikit-image 0.24.0 defaults. Actual segment counts were saved. Images use EXIF transpose, RGB conversion, aspect-preserving maximum side 512, then the same validated CLIP processor. Seed: 42. Model commit: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

## Screening and sampling

All **7,511** official [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) instances were screened: **5,745 correctly ranked**, **1,766 incorrectly ranked**, **0 ties**, accuracy **76.49%**. A seeded, balanced sample of 100 correctly ranked instances was selected before generating explanations. Official category names were preserved. No instance was replaced based on explanation performance.

| Category | Examined | Correct | Incorrect | Accuracy | Selected |
|---|---:|---:|---:|---:|---:|
| add_att | 692 | 480 | 212 | 69.36% | 14 |
| add_obj | 2062 | 1586 | 476 | 76.92% | 14 |
| replace_att | 788 | 631 | 157 | 80.08% | 15 |
| replace_obj | 1652 | 1497 | 155 | 90.62% | 15 |
| replace_rel | 1406 | 975 | 431 | 69.35% | 14 |
| swap_att | 666 | 427 | 239 | 64.11% | 14 |
| swap_obj | 245 | 149 | 96 | 60.82% | 14 |

## Evaluation convention

Both rankings use the same validated Telea inpainting evaluator (`cv2.INPAINT_TELEA`, radius 3), applying cumulative masks to the original image. Only strictly positive SHAP regions are deleted, in descending order, with ascending region-index ties. Pairwise preference is `sigmoid(gamma × (cosine_positive − cosine_negative))`, where pretrained **gamma = 100.000000**. PDAUC integrates preference against actual removed pixel fraction; lower is better.

Per the user’s clarification, after all positive regions are deleted the final preference is held constant to x=1 **solely for AUC integration**. Zero and negative regions are never deleted. With no positive regions, original preference is constant over [0,1]. Unreached 25%/50% thresholds are missing, not extrapolated; rates show coverage and a comparison restricted to instances reached by both methods. The validated global-mean fallback is retained only when Telea has no unmasked context.

## Overall results

| Metric | Pointwise SHAP | CR-SHAP |
|---|---:|---:|
| Mean PDAUC | 0.595358 | 0.548704 |
| Median PDAUC | 0.607296 | 0.552090 |
| Per-instance wins | 29 | 71 |
| No positive attribution | 0 | 0 |

n=100; ties=0. Mean paired PDAUC difference (Pointwise − CR): **+0.046654**; median paired difference: **+0.038533**. Relative mean improvement: **+7.84%**. Positive differences favor CR-SHAP.

Paired two-sided Wilcoxon: statistic **1324.000**, p=**3.63667e-05**. Paired percentile bootstrap 95% CI for the mean difference: **[+0.024429, +0.068630]**, using 10,000 resamples and seed 42.

| Threshold | Pointwise flip rate (reached n) | CR flip rate (reached n) | Common n | Pointwise / CR on common cases |
|---|---:|---:|---:|---:|
| RankFlip@25 | 27.00% (100) | 37.00% (100) | 100 | 27.00% / 37.00% |
| RankFlip@50 | 31.63% (98) | 47.89% (71) | 70 | 34.29% / 48.57% |

| Metric at actual ≥25% deletion | Pointwise | CR-SHAP |
|---|---:|---:|
| Mean margin (lower better) | 0.006686 | 0.004412 |
| Mean margin drop (higher better) | 0.010245 | 0.012519 |

## Category results

| Category | n | Point mean | CR mean | Mean paired difference | Relative improvement | CR / Point wins / ties |
|---|---:|---:|---:|---:|---:|---:|
| add_att | 14 | 0.564657 | 0.522586 | +0.042071 | +7.45% | 10 / 4 / 0 |
| add_obj | 14 | 0.611420 | 0.512175 | +0.099245 | +16.23% | 11 / 3 / 0 |
| replace_att | 15 | 0.621779 | 0.586383 | +0.035396 | +5.69% | 10 / 5 / 0 |
| replace_obj | 15 | 0.691108 | 0.649427 | +0.041682 | +6.03% | 10 / 5 / 0 |
| replace_rel | 14 | 0.550905 | 0.533566 | +0.017339 | +3.15% | 10 / 4 / 0 |
| swap_att | 14 | 0.527291 | 0.482749 | +0.044543 | +8.45% | 9 / 5 / 0 |
| swap_obj | 14 | 0.591617 | 0.544157 | +0.047460 | +8.02% | 11 / 3 / 0 |

| Category | Point / CR RankFlip@25 (coverage) | Point / CR RankFlip@50 (coverage) |
|---|---:|---:|
| add_att | 35.71% (14/14) / 42.86% (14/14) | 30.77% (13/14) / 62.50% (8/14) |
| add_obj | 21.43% (14/14) / 50.00% (14/14) | 28.57% (14/14) / 54.55% (11/14) |
| replace_att | 33.33% (15/15) / 40.00% (15/15) | 21.43% (14/15) / 33.33% (12/15) |
| replace_obj | 20.00% (15/15) / 20.00% (15/15) | 33.33% (15/15) / 40.00% (15/15) |
| replace_rel | 28.57% (14/14) / 50.00% (14/14) | 50.00% (14/14) / 44.44% (9/14) |
| swap_att | 28.57% (14/14) / 28.57% (14/14) | 35.71% (14/14) / 50.00% (6/14) |
| swap_obj | 21.43% (14/14) / 28.57% (14/14) | 21.43% (14/14) / 60.00% (10/14) |

No category-level significance tests were performed on these small strata. Common-case category flip rates are also available in `summary.json`.

## Exploratory margin relationship and visualizations

Spearman correlation between original cosine margin and Pointwise − CR PDAUC difference: rho=0.034839, p=0.730757. This analysis is exploratory; see `margin_relationship.png`.

Fifteen figures were selected strictly by PDAUC difference: five largest improvements, five closest-to-zero remaining differences, and five largest worsenings. IDs and the deterministic selection rule are in `figure_selection.json`. These are outcome-stratified diagnostic examples, not a representative estimate of average visual quality.

## Validation, timing, and interpretation

The five-instance pilot passed before the full run. Five Experiment 2 unit tests passed. All 100 final examples passed sanity checks, including full-coalition scores, correct caption embeddings, shared masks/images, frozen model state, efficiency, strict positive deletion ordering, preference calculation, and Shapley linearity. Maximum absolute linearity error: **1.49e-16**. Experiment 1 file hashes were verified unchanged.

Mean recorded explanation runtime: Pointwise **1.090 s**, CR **1.090 s**. These are shared coalition generation/scoring time plus each target’s own regression time; shared inference actually ran once, not twice. Raw timing fields preserve this distinction.

**Research classification: VERY PROMISING.** All five specified internal criteria are satisfied: relative mean PDAUC improvement is 7.84%, paired p=0.0000364, the 95% paired bootstrap mean-difference CI is positive, CR-SHAP wins 71/100 instances, and mean PDAUC improves in all seven categories. This is exploratory evidence under the stated evaluation convention, not a publication guarantee or proof of general causal faithfulness.

Limitations: one model, one seed, 100 correctly ranked instances drawn from 96 distinct images; the sample is balanced by category, not weighted to benchmark frequency. Repeated images mean instance-level Wilcoxon and bootstrap calculations do not establish independence at image level. Positive-only tail integration and differing positive-area coverage can influence PDAUC. Telea preference deletion is a controlled proxy, not proof of causal faithfulness. Category effects require replication; no parameters were optimized after seeing results.

All raw evidence is retained in `example_*/raw.npz`, `deletion_curves.json`, and associated metadata. Screening evidence and cached normalized embeddings are in `data/sugarcrepe_screening`. Experiment 1 was neither modified nor rerun.
