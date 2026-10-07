# Experiment 3: confirmatory CR-SHAP evaluation on SugarCrepe

The positive-only advantage replicated, and the matched-area endpoint supports the same direction.

This run tests the previously observed advantage under the specified frozen pipeline. Statistical support below means a positive mean paired difference, a two-sided Wilcoxon p < 0.05, and an image-cluster 95% interval above zero; these are reported explicitly rather than as proof of causal faithfulness.

## Frozen design and sample

Official SugarCrepe screening reproduced **7,511 instances, 5,745 correctly ranked, 1,766 incorrectly ranked, and zero ties** (76.49% pair accuracy). The new screening recomputed embeddings and scores rather than using the old counts.

Sampling seed 2026 initially selected **700 correctly ranked instances**, 100 in every official category. All **100 Experiment 2 example IDs** were excluded, with no eligible-pool shortages or reintroductions. The final sample contains **699 instances: 99 swap_att and 100 in each other category** after the authorized technical exclusion described below. Sampling used at most one instance per image within each category: uniformly sampled image IDs, then a uniformly sampled eligible instance for each image. Thus sampling is uniform over images within category, not uniform over caption instances. Cross-category image reuse was allowed as specified.

Protocol amendment: the selected instance `swap_att:39` produced six regions under frozen SLIC, which permit only 64 unique binary coalitions. The unchanged sampler requires 128 unique coalitions. After an outcome-independent check of all 700 segmentations, the original run was stopped at 55 completed instances. The user authorized fixing and running after the proposed single technical exclusion. This instance was excluded without replacement; no method settings changed. The exclusion was made after some outcomes had been computed, but its sole criterion was coalition feasibility. Original selections and partial results remain archived in `results/experiment_3_stopped_001`, and the original pilot in `results/experiment_3_pilot_original`. The 55 independently verified completed instances were reused verbatim; the remaining instances were evaluated under the same frozen configuration.

There are **596 unique images**, **94 images used more than once**, and at most **3 instances for one image**. Exclusion applies to instance IDs, not all images seen in Experiment 2; 41 images also occurred there. The image-cluster bootstrap accounts for image repetition within Experiment 3.

Frozen `openai/clip-vit-base-patch32` at commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, eval mode, gamma 100.0. Images use EXIF transpose, RGB, aspect-preserving maximum side 512, then the validated CLIP processor. SLIC uses 16 requested segments, compactness 30, start label 0, channel axis −1, and scikit-image 0.24.0 defaults. The unchanged constrained Kernel SHAP estimator uses 128 unique coalitions and seed 42 with mean RGB masking. One image embedding supplies both caption scores per coalition. The same positive and official negative captions are used throughout. No model training or parameter tuning occurred.

## Endpoints and paired inference

| Endpoint (lower is better) | Pointwise mean / median | CR mean / median | Mean difference | CR wins | Wilcoxon p | Image-cluster 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Primary: positive-only PDAUC | 0.613365 / 0.610474 | 0.559526 / 0.557078 | +0.053838 | 461/699 | 8.58125e-24 | [+0.043704, +0.064042] |
| Robustness: MatchedArea-PDAUC-50 | 0.670330 / 0.667499 | 0.625471 / 0.621317 | +0.044858 | 452/699 | 7.80693e-23 | [+0.036384, +0.053332] |

Primary relative mean improvement: **8.78%**; median paired difference +0.034783; Pointwise wins 238, ties 0. Wilcoxon statistic 68616. Paired instance-bootstrap CI [+0.043857, +0.064055].

Matched-area relative mean improvement: **6.69%**; median paired difference +0.026777; Pointwise wins 246, ties 1. Wilcoxon statistic 69553. Paired instance-bootstrap CI [+0.036249, +0.053557].

Both bootstraps use 10,000 resamples, seed 42, and percentile intervals. The cluster bootstrap draws the original number of image IDs with replacement, includes every instance belonging to each drawn image (including repeats of that cluster), and computes an instance-weighted mean difference. Cluster intervals are the preferred uncertainty estimates. Wilcoxon is the requested paired instance test and does not itself model image clustering.

Positive-only PDAUC reproduces Experiment 2: delete strictly positive regions in descending SHAP order, index ties ascending, using Telea radius 3 on the original image with each cumulative mask. Extend the final preference horizontally to x=1 for integration only; no positives means a constant original preference. Zero and negative regions are never removed in this evaluation.

Matched-area deletion ranks **all signed SHAP values**, descending with index ties ascending. For each 10–50% target it uses the first whole-region state reaching or exceeding the target. MatchedArea-PDAUC-50 integrates the original preference and these five observed states against their actual pixel fractions, dividing by the final fraction. Repeated crossing states contribute zero-width intervals. There is no extrapolation to 100%. Telea and the validated fully masked global-mean fallback remain unchanged.

## Fixed removal budgets

| Budget | Actual area Point / CR | Preference Point / CR | Preference difference | Margin Point / CR | Margin drop Point / CR | Drop difference | Rank flip Point / CR | Exact McNemar p |
|---|---|---|---:|---|---|---:|---|---:|
| 10% | 14.862% / 14.674% | 0.702813 / 0.669187 | +0.033625 | +0.013408 / +0.011071 | 0.006033 / 0.008371 | +0.002337 | 17.31% / 20.74% | 0.0196902 |
| 20% | 24.408% / 24.399% | 0.672548 / 0.623305 | +0.049243 | +0.011019 / +0.007806 | 0.008422 / 0.011636 | +0.003214 | 21.75% / 28.47% | 2.72042e-05 |
| 30% | 34.406% / 34.273% | 0.638723 / 0.591880 | +0.046843 | +0.008678 / +0.005576 | 0.010763 / 0.013865 | +0.003102 | 26.61% / 32.90% | 0.000794063 |
| 40% | 44.460% / 44.110% | 0.614578 / 0.552001 | +0.062577 | +0.006925 / +0.003180 | 0.012517 / 0.016262 | +0.003745 | 30.47% / 40.49% | 7.29263e-07 |
| 50% | 53.957% / 54.005% | 0.590476 / 0.511411 | +0.079065 | +0.005436 / +0.000840 | 0.014006 / 0.018602 | +0.004596 | 34.19% / 47.64% | 9.07787e-13 |

Every instance reaches every matched-area budget. Whole-region overshoot means actual areas can differ between methods; this is the specified approximate budget matching, not exact equal-pixel removal.

| Budget | Neither flipped | Pointwise only | CR only | Both flipped | Preference cluster CI | Margin-drop cluster CI |
|---|---:|---:|---:|---:|---|---|
| 10% | 517 | 37 | 61 | 84 | [+0.023541, +0.044047] | [+0.001728, +0.002961] |
| 20% | 462 | 38 | 85 | 114 | [+0.037534, +0.060914] | [+0.002465, +0.003954] |
| 30% | 408 | 61 | 105 | 125 | [+0.032588, +0.061105] | [+0.002250, +0.003967] |
| 40% | 352 | 64 | 134 | 149 | [+0.047358, +0.077652] | [+0.002830, +0.004668] |
| 50% | 324 | 42 | 136 | 197 | [+0.063749, +0.094350] | [+0.003687, +0.005503] |

The exact two-sided McNemar test is the binomial test with probability 0.5 on the discordant paired outcomes. It operates on instance pairs and does not adjust for shared-image dependence. Fixed-budget and category tests are secondary and unadjusted for multiple comparisons; see `summary.json` for all paired bootstrap intervals, Wilcoxon statistics, and p-values.

## Category results

| Category | n | Positive PDAUC Point / CR | Positive difference | Matched PDAUC Point / CR | Matched difference |
|---|---:|---|---:|---|---:|
| add_att | 100 | 0.611596 / 0.576638 | +0.034958 | 0.643938 / 0.622425 | +0.021513 |
| add_obj | 100 | 0.616739 / 0.572724 | +0.044015 | 0.660766 / 0.632470 | +0.028296 |
| replace_att | 100 | 0.667657 / 0.582888 | +0.084769 | 0.733275 / 0.663987 | +0.069289 |
| replace_obj | 100 | 0.666081 / 0.590687 | +0.075393 | 0.761092 / 0.698412 | +0.062680 |
| replace_rel | 100 | 0.592153 / 0.544765 | +0.047389 | 0.652252 / 0.607108 | +0.045144 |
| swap_att | 99 | 0.576744 / 0.540541 | +0.036204 | 0.622224 / 0.595059 | +0.027165 |
| swap_obj | 100 | 0.562218 / 0.508252 | +0.053965 | 0.618278 / 0.558536 | +0.059743 |

## Validation, interpretation, and evidence

Three new unit tests passed. A seven-instance pilot covered every category and passed before the full run. All 699 final instances passed checks; independent reconstruction validated 89,472 coalition masks/images by hashes plus saved deletion masks, rankings, first crossings, and metric formulas. Maximum Shapley linearity error: 1.91e-16. All 2,026 protected Experiment 1/2 source and result files remained unchanged.

Positive-only summaries: mean positive area Pointwise 75.317%, CR 58.870%; mean positive-region count Pointwise 9.66, CR 7.75; no-positive cases Pointwise 0, CR 0.

The comparison concerns CLIP preference under Telea perturbations. It does not establish causal faithfulness beyond that evaluator. Conditioning on correctly ranked examples, equal category weighting, approximate Kernel SHAP, whole-region budget overshoot, one model and sampling seed, and residual image dependence limit generalization. The matched-area evaluation addresses the positive-only constant-tail concern but still uses different removal states and may end slightly beyond 50%. Excluding discovery instance IDs does not guarantee disjoint images. No outcome-based sample replacement or method tuning was performed.

Raw evidence is in `example_000` through `example_698`: original images, segment maps, shared binary coalitions and hashes, positive/negative coalition scores, all three SHAP vectors, regression diagnostics, positive-only curves, full signed orders, actual cumulative matched masks, cosine scores, margins, preferences, and validation checks. The complete per-instance table is `results.csv`; inference time is recorded once as shared cost, with Pointwise and CR regression times separate. Source revision, selection, package versions, screening counts, hashes, and integrity evidence are saved alongside the metrics.

The supplied request ended at “exact McNem” in section 30; this was interpreted as exact McNemar. No additional requirements beyond the supplied text were assumed.
