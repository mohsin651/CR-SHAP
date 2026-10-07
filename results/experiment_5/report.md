# Experiment 5: external Grad-ECLIP baseline and contrastive objective

Yes: the contrastive objective improves the primary endpoint with paired statistical support.

CR-SHAP has lower observed mean matched-area PDAUC than Contrastive Grad-ECLIP.

## Scientific question and implementation

Does explaining relative caption preference help an established gradient explanation family, or is the observed benefit specific to SHAP? Four configurations share the exact 300-image Experiment 4 sample and evaluator. SHAP results are reused verbatim.

This is the **model-preserving multi-head Grad-ECLIP variant**, documented in Appendix C of the [ICML 2024 paper](https://proceedings.mlr.press/v235/zhao24p.html). The [official notebook](https://github.com/Cyang-Zhao/Grad-Eclip/blob/e370e6cb194faf2020f5d1ed268f9d57e91a38e6/grad_eclip_image.ipynb) is pinned at commit `e370e6cb194faf2020f5d1ed268f9d57e91a38e6` and archived locally. Its default final-block single-head rewrite changes the model function; using that default would violate this experiment’s frozen-model requirement. Consequently these results must not be described as a numerical reproduction of the default single-head demo.

The target is visual transformer block 11 of 12. Its twelve original attention heads remain intact. The activation is the concatenated CLS attention context before its output projection. Channel weights are its target-score gradient. Spatial weights follow the notebook: min-max-normalized cosine of the unscaled CLS query and each patch key over concatenated channels. Patch relevance is ReLU of the channel sum of gradient × patch value × spatial weight. This preserves Grad-ECLIP’s value features and spatial/channel weighting, rather than substituting Grad-CAM.

Both methods share one forward graph. Pointwise targets raw cosine s(I,T+); contrastive targets raw cosine s(I,T+) − s(I,T−). The scalar is changed before autograd and ReLU; two positive heatmaps are not subtracted. Gradient linearity is checked before ReLU. All parameters remain frozen and eval-mode; there is no optimizer or training loop.

The native 7×7 map is bilinearly interpolated to the processor’s 224×224 crop, placed back into its actual aspect-preserving resized canvas, and resized to the saved working image, with align_corners=False. Pixels outside the visible center crop receive zero relevance. No map normalization is applied for evaluation. SLIC relevance is the mean over all pixels in each saved region. Display heatmaps alone may be normalized.

## Frozen sample and evaluation

All 300 image IDs, original images, positive/negative captions, caption IDs/ownership, original cosine scores, margins, and SLIC maps are unchanged from Experiment 4. The strongest other-image caption is fixed from the complete 5,000-caption retrieval pool; no examples or negatives were replaced.

CLIP: `openai/clip-vit-base-patch32`, commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. EXIF transpose, RGB, aspect-preserving maximum side 512, then the same processor and tokenizer. SLIC was reused, not recomputed or tuned. Seeds: 42 for deterministic runtime and bootstrap; the existing selection seed remains 2026.

The primary endpoint is MatchedArea-PDAUC-50. All region scores are ranked descending with ascending-index ties. Whole regions are removed until the first crossing of 10/20/30/40/50% actual pixel area. Telea radius 3 is applied to the original image for each cumulative mask, with the unchanged global-mean fallback for full removal. Preference is sigmoid(gamma × raw cosine margin), gamma from the original model, approximately 100. Trapezoidal AUC uses the original state and five observed crossing states, divides by final area, and has no extrapolation. Duplicate crossings contribute zero-width intervals.

PositiveOnly-PDAUC is secondary: strictly positive region scores are deleted, then the final preference is held constant to 100% for integration only. Grad-ECLIP’s official ReLU yields nonnegative maps, so this endpoint reflects positive spatial support, not signed SHAP contributions; its cross-family interpretation is limited.

## Complete 2×2 headline comparison

| Method | Objective | MatchedArea-PDAUC-50 mean / median | Relative improvement vs Pointwise SHAP | Wins / losses / ties vs Pointwise SHAP |
|---|---|---|---:|---|
| Pointwise SHAP | s(I,T+) | 0.681292 / 0.750801 | 0.00% | 0 / 0 / 300 |
| CR-SHAP | s(I,T+) − s(I,T−) | 0.618664 / 0.651444 | 9.19% | 193 / 107 / 0 |
| Grad-ECLIP (multi-head) | s(I,T+) | 0.665729 / 0.729070 | 2.28% | 169 / 131 / 0 |
| Contrastive Grad-ECLIP (multi-head) | s(I,T+) − s(I,T−) | 0.635334 / 0.688040 | 6.75% | 180 / 120 / 0 |

## Paired primary comparisons

| Comparison (first → second) | First mean / median | Second mean / median | Mean / median difference | Relative improvement | Second / first wins / ties | Wilcoxon statistic / p | Paired mean-difference 95% CI |
|---|---|---|---|---:|---|---|---|
| A_SHAP_objective: Pointwise SHAP → CR-SHAP | 0.681292 / 0.750801 | 0.618664 / 0.651444 | +0.062628 / +0.030781 | 9.19% | 193 / 107 / 0 | 13475 / 1.43468e-09 | [+0.043120, +0.081947] |
| B_GradECLIP_objective: Grad-ECLIP (multi-head) → Contrastive Grad-ECLIP (multi-head) | 0.665729 / 0.729070 | 0.635334 / 0.688040 | +0.030396 / +0.014286 | 4.57% | 182 / 112 / 6 | 15715 / 4.30806e-05 | [+0.015473, +0.045489] |
| C_pointwise_families: Pointwise SHAP → Grad-ECLIP (multi-head) | 0.681292 / 0.750801 | 0.665729 / 0.729070 | +0.015563 / +0.007960 | 2.28% | 169 / 131 / 0 | 19880 / 0.073103 | [-0.004690, +0.036285] |
| D_contrastive_families: CR-SHAP → Contrastive Grad-ECLIP (multi-head) | 0.618664 / 0.651444 | 0.635334 / 0.688040 | -0.016670 / -0.004319 | -2.69% | 137 / 163 / 0 | 19702 / 0.0560615 | [-0.034472, +0.000974] |

n=300 unique paired query images in every comparison. Positive AUC/preference differences are first minus second and favor the second method; positive margin-drop differences are second minus first. Tests are paired two-sided Wilcoxon; CIs use 10,000 paired percentile bootstrap resamples, seed 42. B is the key new objective comparison. A exactly reproduces the frozen Experiment 4 numbers. C/D and fixed budgets are secondary; p-values are unadjusted and cannot be treated as independent evidence.

## Secondary positive-only results

| Comparison | First mean / median | Second mean / median | Difference | Relative improvement | Second / first wins / ties | Wilcoxon p | Paired 95% CI |
|---|---|---|---:|---:|---|---:|---|
| A_SHAP_objective | 0.587933 / 0.613426 | 0.514820 / 0.510955 | +0.073113 | 12.44% | 200 / 100 / 0 | 1.96448e-10 | [+0.051030, +0.095774] |
| B_GradECLIP_objective | 0.581047 / 0.598519 | 0.550427 / 0.561267 | +0.030620 | 5.27% | 173 / 125 / 2 | 6.3369e-06 | [+0.018028, +0.042900] |
| C_pointwise_families | 0.587933 / 0.613426 | 0.581047 / 0.598519 | +0.006886 | 1.17% | 156 / 144 / 0 | 0.588291 | [-0.011745, +0.025736] |
| D_contrastive_families | 0.514820 / 0.510955 | 0.550427 / 0.561267 | -0.035607 | -6.92% | 125 / 175 / 0 | 0.000840845 | [-0.056194, -0.015084] |

## Secondary fixed budgets and RankFlip

| Method | Budget | Mean actual area | Mean preference | Mean margin drop | RankFlip |
|---|---:|---:|---:|---:|---:|
| Pointwise SHAP | 10% | 15.67% | 0.714802 | 0.012825 | 22.67% |
| Pointwise SHAP | 20% | 24.79% | 0.676075 | 0.016783 | 28.00% |
| Pointwise SHAP | 30% | 34.64% | 0.628664 | 0.021329 | 35.33% |
| Pointwise SHAP | 40% | 44.39% | 0.601392 | 0.024881 | 35.33% |
| Pointwise SHAP | 50% | 54.50% | 0.562834 | 0.027830 | 41.67% |
| CR-SHAP | 10% | 15.15% | 0.681972 | 0.017035 | 29.00% |
| CR-SHAP | 20% | 24.75% | 0.607422 | 0.023994 | 37.67% |
| CR-SHAP | 30% | 34.64% | 0.549038 | 0.028691 | 44.67% |
| CR-SHAP | 40% | 43.99% | 0.495695 | 0.032996 | 49.33% |
| CR-SHAP | 50% | 54.23% | 0.457053 | 0.036323 | 54.00% |
| Grad-ECLIP (multi-head) | 10% | 14.87% | 0.716207 | 0.013714 | 23.67% |
| Grad-ECLIP (multi-head) | 20% | 24.99% | 0.659667 | 0.018829 | 30.67% |
| Grad-ECLIP (multi-head) | 30% | 34.65% | 0.613681 | 0.022645 | 36.33% |
| Grad-ECLIP (multi-head) | 40% | 44.76% | 0.551577 | 0.027694 | 43.67% |
| Grad-ECLIP (multi-head) | 50% | 54.72% | 0.530657 | 0.029609 | 45.33% |
| Contrastive Grad-ECLIP (multi-head) | 10% | 15.03% | 0.691892 | 0.015700 | 28.00% |
| Contrastive Grad-ECLIP (multi-head) | 20% | 25.36% | 0.612822 | 0.022888 | 37.67% |
| Contrastive Grad-ECLIP (multi-head) | 30% | 34.81% | 0.572926 | 0.026716 | 41.67% |
| Contrastive Grad-ECLIP (multi-head) | 40% | 44.73% | 0.537083 | 0.029644 | 44.00% |
| Contrastive Grad-ECLIP (multi-head) | 50% | 54.74% | 0.483418 | 0.033925 | 50.00% |

| Comparison | Budget | Neither | First only | Second only | Both | Exact McNemar p | Preference diff / CI | Margin-drop diff / CI |
|---|---:|---:|---:|---:|---:|---:|---|---|
| A_SHAP_objective | 10% | 198 | 15 | 34 | 53 | 0.00939924 | +0.032830 / [+0.007715, +0.056651] | +0.004210 / [+0.002306, +0.006188] |
| A_SHAP_objective | 20% | 171 | 16 | 45 | 68 | 0.000264279 | +0.068653 / [+0.039671, +0.098211] | +0.007211 / [+0.005010, +0.009438] |
| A_SHAP_objective | 30% | 145 | 21 | 49 | 85 | 0.00109324 | +0.079626 / [+0.045610, +0.113484] | +0.007361 / [+0.004758, +0.009966] |
| A_SHAP_objective | 40% | 135 | 17 | 59 | 89 | 1.39687e-06 | +0.105697 / [+0.072354, +0.139610] | +0.008116 / [+0.005626, +0.010648] |
| A_SHAP_objective | 50% | 112 | 26 | 63 | 99 | 0.000109926 | +0.105781 / [+0.069672, +0.140974] | +0.008493 / [+0.005859, +0.011102] |
| B_GradECLIP_objective | 10% | 204 | 12 | 25 | 59 | 0.047031 | +0.024315 / [+0.001795, +0.047819] | +0.001986 / [+0.000197, +0.003838] |
| B_GradECLIP_objective | 20% | 177 | 10 | 31 | 82 | 0.00145049 | +0.046846 / [+0.023218, +0.070996] | +0.004059 / [+0.002021, +0.006094] |
| B_GradECLIP_objective | 30% | 159 | 16 | 32 | 93 | 0.0293049 | +0.040754 / [+0.013636, +0.067685] | +0.004071 / [+0.001916, +0.006259] |
| B_GradECLIP_objective | 40% | 141 | 27 | 28 | 104 | 1 | +0.014493 / [-0.014801, +0.044178] | +0.001950 / [-0.000329, +0.004292] |
| B_GradECLIP_objective | 50% | 128 | 22 | 36 | 114 | 0.0869489 | +0.047239 / [+0.017680, +0.076563] | +0.004317 / [+0.002084, +0.006538] |
| C_pointwise_families | 10% | 211 | 18 | 21 | 50 | 0.749259 | -0.001405 / [-0.026782, +0.024647] | +0.000889 / [-0.001380, +0.003110] |
| C_pointwise_families | 20% | 184 | 24 | 32 | 60 | 0.349682 | +0.016408 / [-0.015402, +0.047704] | +0.002046 / [-0.000657, +0.004722] |
| C_pointwise_families | 30% | 158 | 33 | 36 | 73 | 0.809949 | +0.014983 / [-0.020022, +0.051085] | +0.001316 / [-0.001515, +0.004190] |
| C_pointwise_families | 40% | 145 | 24 | 49 | 82 | 0.00462629 | +0.049815 / [+0.014227, +0.086825] | +0.002813 / [-0.000039, +0.005841] |
| C_pointwise_families | 50% | 132 | 32 | 43 | 93 | 0.248046 | +0.032177 / [-0.002806, +0.066505] | +0.001779 / [-0.000996, +0.004553] |
| D_contrastive_families | 10% | 188 | 28 | 25 | 59 | 0.783846 | -0.009920 / [-0.034708, +0.015701] | -0.001335 / [-0.003072, +0.000471] |
| D_contrastive_families | 20% | 156 | 31 | 31 | 82 | 1 | -0.005400 / [-0.033838, +0.023224] | -0.001105 / [-0.003131, +0.000918] |
| D_contrastive_families | 30% | 135 | 40 | 31 | 94 | 0.342471 | -0.023888 / [-0.056353, +0.008991] | -0.001975 / [-0.004230, +0.000326] |
| D_contrastive_families | 40% | 127 | 41 | 25 | 107 | 0.0640175 | -0.041388 / [-0.073317, -0.009193] | -0.003352 / [-0.005641, -0.001039] |
| D_contrastive_families | 50% | 104 | 46 | 34 | 116 | 0.218518 | -0.026365 / [-0.063643, +0.011032] | -0.002398 / [-0.005119, +0.000331] |

Exact two-sided McNemar uses a binomial test on the discordant paired outcomes. Full paired preference/margin-drop Wilcoxon statistics, medians, wins and CIs are in `summary.json`.

## Technical validation and reproducibility

Three new unit tests validate the official notebook formula, unchanged multi-head attention, inverse crop geometry, region means, and pre-ReLU contrastive linearity. A five-image technical pilot passed before the full run. All 300 examples passed native-score agreement (<1e-6), gradient linearity (<1e-6), repeated-gradient/no-accumulation, finite maps, expected spatial dimensions, and nonzero gradient checks. Maximum native score discrepancy: 1.49e-07; maximum gradient linearity discrepancy: 6.75e-09.

Full model state hashes match before and after explanation generation. Independent saved-evidence verification checked maps, mean region scores, deletion rankings and cumulative masks, first crossings, preference/AUC formulas, paired inference, and verbatim SHAP reuse. All 11,987 protected prior source/result/data files stayed unchanged. No SHAP coalition inference was recomputed.

Package, PyTorch, CUDA, cuDNN and GPU versions are in `environment.json`; parameters and source hashes in `config.json` and `model_integrity.json`. Per-example `raw.npz` preserves Q/K/V, spatial weights, objective gradients, pre-ReLU/native patch maps, working-image pixel maps, SLIC scores, and matched cumulative masks. Separate JSON files preserve crop geometry, curves and technical checks.

## Runtime

Both gradient objectives together: 25.57s explanation/technical-check time (0.0852s per image), plus 536.87s shared evaluator cost. Existing SHAP shared coalition inference: 351.66s, plus 0.11s for the two target regressions.

Gradient timing includes the native-forward comparison, negative-gradient and repeat-gradient validations, and map projection; SHAP inference includes coalition masking/scoring. These are documented workloads, not identical benchmark scopes. Timing excludes model loading, hash snapshots, report generation, and disk serialization. Identical evaluator images are cached across gradient methods and endpoints.

## Interpretation and limitations

Yes: the contrastive objective improves the primary endpoint with paired statistical support.

CR-SHAP has lower observed mean matched-area PDAUC than Contrastive Grad-ECLIP.

Key B: mean difference +0.030396, relative mean improvement 4.57%, Wilcoxon p=4.30806e-05, paired CI [+0.015473, +0.045489]. Objective-controlled D: first=CR-SHAP, second=Contrastive Grad-ECLIP, mean difference -0.016670, p=0.0560615, CI [-0.034472, +0.000974].

If both A and B support a positive difference, this supports contrastive preference as a useful principle across these two explanation families under this evaluator. Otherwise the gradient comparison does not establish the same benefit; absence of statistical support is not proof of no effect. A family difference is specific to the chosen published variant and evaluator and cannot prove universal SHAP superiority.

The frozen-model multi-head variant differs from the authors’ recommended single-head demo and is labeled throughout. One final layer, coarse 7×7 resolution, positive ReLU support, mean pixel relevance versus SHAP region contributions, inverse-crop zero padding, whole-region budget overshoot, correctly-ranked-only conditioning, semantic overlap among natural false captions, shared candidate captions, and one model/sample seed limit interpretation. Telea removal tests model preference under perturbations rather than causal faithfulness. No outcome-based tuning, resampling or evaluator changes occurred.

## Failure cases and diagnostic examples

Zero native ReLU maps: Pointwise 0, Contrastive 0. Finite zero maps in the full sample are preserved and evaluated with deterministic region-index ties, never replaced. Largest objective worsenings and their captions are in `failure_cases.json`.

Fifteen figures show the five largest gradient objective improvements, five closest-to-zero unused differences, and five smallest differences. These outcome-stratified diagnostic examples are not representative estimates of population performance. Selection and exact IDs are saved.

![improvement: image 4846324908](figures/improvement_148.png)

![improvement: image 3903017514](figures/improvement_237.png)

![improvement: image 1801663973](figures/improvement_089.png)

![improvement: image 16151663](figures/improvement_252.png)

![improvement: image 246231741](figures/improvement_161.png)

![near_zero: image 166283675](figures/near_zero_031.png)

![near_zero: image 314739483](figures/near_zero_081.png)

![near_zero: image 3970114165](figures/near_zero_176.png)

![near_zero: image 42348693](figures/near_zero_014.png)

![near_zero: image 4735200580](figures/near_zero_197.png)

![worsening: image 416992999](figures/worsening_145.png)

![worsening: image 3039200576](figures/worsening_041.png)

![worsening: image 2870426310](figures/worsening_122.png)

![worsening: image 4773842539](figures/worsening_108.png)

![worsening: image 4830651041](figures/worsening_116.png)

Experiment 5 is complete. No Experiment 6 or other follow-up was launched. Human review is required before further work.
