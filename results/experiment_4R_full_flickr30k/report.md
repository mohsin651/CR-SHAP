# Experiment4R: full eligible Flickr30k robustness analysis

CR-SHAP has lower observed mean matched-area PDAUC across all793 originally eligible queries.

This is a post-specified full eligible-test-population robustness analysis. The original pre-specified/random-sample Experiment4 result remains unchanged and is reported separately. It is not replaced by this extension.

## Exact population and integrity

Original screening:1000 Karpathy test images and5000 owned captions,793 strictly correctly ranked queries,207 incorrect,0 exact ties. Original300 and remaining493 are disjoint and their union is exactly793. Canonical eligible-ID checksum: `c4bbdc80282a0ffef053a403c62b7c7812dab59f112dc4f4db49bb53ef22b751` (SHA256 of compact UTF8 JSON of lexicographically sorted IDs). No new retrieval screening, model-defined captions, eligibility, sampling seed or reserve list was introduced.

Intended census=793; original reused=300; new intended=493; new evaluable=493; total evaluable=793; technical failures=0. Every failure and its exact phase/reason is retained in `technical_failures.json`. No failed query was replaced, and no outcome-based exclusion was used.

All original300 artifact files were hashed before reuse and checked again afterwards. Five original examples were independently reconstructed using the existing verifier with its output redirected into this new directory. The original300 per-example metrics and complete statistical summary reproduce exactly in their original order. Their images, SLIC, coalitions, score caches, SHAP vectors and deletion curves are referenced in place, never recomputed or rewritten.

## Unchanged Experiment4 protocol

Model `openai/clip-vit-base-patch32`, commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, frozen eval mode. T+ and T−, caption IDs/owners, and scores come only from the original B/32 screening. B/16 competitors and embeddings are not used. Gamma remains100 from the same model logit scale. All method settings are preserved verbatim in `original_protocol_config.json`; its sampling description refers to the original300 study, while `scope.json` defines this census extension.

EXIF transpose, RGB, aspect-preserving max-side512, unchanged CLIP processor/tokenizer. SLIC:16 requested regions, compactness30, start_label0, channel_axis−1, scikit-image0.24.0 defaults. Mean-RGB masking;128 unique shared coalitions seed42 with mandatory endpoints/singletons/complements; unchanged constrained Kernel SHAP. One normalized coalition image embedding provides both caption cosine scores, with negative-caption regression validating Shapley linearity. All new feasibility checks finished before new explanation outcomes.

Primary MatchedArea-PDAUC-50: all signed region values descending, ascending region-index ties; first whole-region crossings at10/20/30/40/50%; actual removed area; Telea radius3 on the original image and unchanged all-mask RGB-mean fallback. Preference=sigmoid(gamma × cosine margin). Integrate original and five crossing states, divide by final actual fraction, retain repeated crossings with zero-width intervals, no extrapolated tail. Secondary positive-only PDAUC preserves the original strictly-positive ranking and constant terminal integration tail to100%.

## Critical three-way primary comparison

| Population | n | Pointwise mean / median | CR mean / median | Absolute mean / median difference | Relative improvement | CR / Point wins / ties | CR win rate | Paired95% CI | Wilcoxon statistic / p |
|---|---:|---|---|---|---:|---|---:|---|---|
| Original sampled Experiment4 | 300 | 0.681292 / 0.750801 | 0.618664 / 0.651444 | +0.062628 / +0.030781 | 9.19% | 193 / 107 / 0 | 64.33% | [+0.043120, +0.081947] | 13475 / 1.43468e-09 |
| Previously unevaluated remainder | 493 | 0.697558 / 0.784139 | 0.626891 / 0.670561 | +0.070666 / +0.040118 | 10.13% | 343 / 148 / 2 | 69.57% | [+0.057230, +0.084226] | 29196 / 3.48106e-23 |
| Complete eligible population | 793 | 0.691404 / 0.769122 | 0.623779 / 0.657747 | +0.067625 / +0.036065 | 9.78% | 536 / 255 / 2 | 67.59% | [+0.056800, +0.079036] | 82358 / 7.18527e-31 |

Full minus original absolute paired effect: +0.004997; relative improvement change: +0.59 percentage points. The original300 are contained in the census; no independent-samples comparison or inappropriate p-value between them was calculated.

The remainder is previously unevaluated under the Experiment4 B/32 protocol, not a newly randomized confirmation sample. No settings were changed after viewing its result.

## Finite population versus broader inference

For a complete793-query census, the observed finite-population mean and effect are directly known; a bootstrap CI is not needed to estimate that already-observed mean. For comparability, paired two-sided Wilcoxon and10000 paired percentile bootstrap resamples, seed42, describe paired variability and potential broader generalization uncertainty under exchangeability assumptions. They are not a probability-sampling guarantee about other datasets, models, or all retrieval queries. Eligibility conditions on correct retrieval, and candidate-caption sharing also limits simple generalization. If failures exist, descriptive results apply only to the stated evaluable population.

## Secondary positive-only PDAUC

| Population | Point mean / median | CR mean / median | Mean difference | Relative improvement | CR / Point wins / ties | 95% CI | Wilcoxon p |
|---|---|---|---:|---:|---|---|---:|
| Original300 | 0.587933 / 0.613426 | 0.514820 / 0.510955 | +0.073113 | 12.44% | 200 / 100 / 0 | [+0.051030, +0.095774] | 1.96448e-10 |
| Remainder | 0.606067 / 0.624559 | 0.539453 / 0.556080 | +0.066614 | 10.99% | 326 / 167 / 0 | [+0.050515, +0.082534] | 1.0676e-14 |
| Full eligible evaluable | 0.599207 / 0.622237 | 0.530134 / 0.538314 | +0.069072 | 11.53% | 526 / 267 / 0 | [+0.056082, +0.082511] | 1.40867e-23 |

## Fixed-budget and RankFlip robustness

| Population | Budget | Actual area Point / CR | Preference Point / CR | Margin drop Point / CR | RankFlip Point / CR | Neither / Point only / CR only / Both | Exact McNemar p |
|---|---:|---|---|---|---|---|---:|
| Full eligible evaluable | 10% | 15.82% / 15.21% | 0.728704 / 0.676007 | 0.012533 / 0.018214 | 22.70% / 29.76% | 525 / 32 / 88 / 148 | 3.18106e-07 |
| Full eligible evaluable | 20% | 24.97% / 24.65% | 0.679784 / 0.609171 | 0.017210 / 0.024278 | 27.74% / 36.57% | 466 / 37 / 107 / 183 | 4.44435e-09 |
| Full eligible evaluable | 30% | 34.68% / 34.62% | 0.639822 / 0.557960 | 0.021009 / 0.028299 | 33.29% / 43.88% | 403 / 42 / 126 / 222 | 6.00951e-11 |
| Full eligible evaluable | 40% | 44.34% / 44.24% | 0.609940 / 0.509401 | 0.024578 / 0.032315 | 35.44% / 48.55% | 363 / 45 / 149 / 236 | 3.30282e-14 |
| Full eligible evaluable | 50% | 54.34% / 54.24% | 0.577030 / 0.473619 | 0.027226 / 0.035498 | 40.48% / 52.33% | 318 / 60 / 154 / 261 | 1.03121e-10 |
| Previously unevaluated remainder | 10% | 15.91% / 15.24% | 0.737163 / 0.672377 | 0.012355 / 0.018932 | 22.72% / 30.22% | 327 / 17 / 54 / 95 | 1.25268e-05 |
| Previously unevaluated remainder | 20% | 25.08% / 24.59% | 0.682041 / 0.610236 | 0.017470 / 0.024450 | 27.59% / 35.90% | 295 / 21 / 62 / 115 | 7.50533e-06 |
| Previously unevaluated remainder | 30% | 34.70% / 34.61% | 0.646611 / 0.563389 | 0.020815 / 0.028061 | 32.05% / 43.41% | 258 / 21 / 77 / 137 | 1.08925e-08 |
| Previously unevaluated remainder | 40% | 44.31% / 44.40% | 0.615142 / 0.517741 | 0.024394 / 0.031901 | 35.50% / 48.07% | 228 / 28 / 90 / 147 | 8.91405e-09 |
| Previously unevaluated remainder | 50% | 54.25% / 54.24% | 0.585669 / 0.483700 | 0.026858 / 0.034996 | 39.76% / 51.32% | 206 / 34 / 91 / 162 | 3.4738e-07 |

All budgets are reported without cherry-picking. Exact two-sided McNemar is the binomial test on discordant paired flips. Full paired preference/margin-drop/positive-only analyses remain in `summary.json`; these secondary p-values are unadjusted. Positive AUC/preference differences mean Pointwise minus CR; positive margin-drop and flip-rate differences mean CR minus Pointwise.

## Required answers

**1. Does CR have lower matched-area PDAUC across the original eligible population?** Yes, descriptively: Pointwise 0.691404, CR 0.623779, paired mean difference +0.067625, n=793. Intended n=793; any missing coverage is explicitly reported above.

**2. Full effect versus the original+9.19%?** The full-evaluable effect is +9.78%, a change of +0.59 percentage points; absolute effect changed by +0.004997.

**3. Does the advantage persist on the previously unevaluated remainder?** The observed mean still favors CR: +10.13%, n=493, CI [+0.057230, +0.084226], Wilcoxon p=3.48106e-23. This is a remainder diagnostic, not an independently randomized confirmation.

**4. Does RankFlip retain the qualitative advantage?** CR has a higher observed flip rate at every specified budget in the full-evaluable population. Exact McNemar results are provided separately for all budgets.

**5. Was the original sample reasonably representative?** The positive direction persists across the full-evaluable population and remainder. The magnitude comparison above describes how much the original estimate differs; this is not a formal equivalence test or proof of representativeness.

## Validation, cost and artifacts

Both newly computed pilot and full remainder were independently checked. Verified new queries=493; coalition masks/perturbations=63,104; maximum linearity error=2.36e-16. Full model state hashes match before/after. Original300 reused metrics and summary match exactly; all prior source/results are protected by before/after hashes.

New-computation runtime seconds: {'new_shared_coalition_inference_seconds': 648.6499893999771, 'new_point_regression_seconds': 0.1460416999689187, 'new_cr_regression_seconds': 0.06731159999799274, 'new_evaluation_seconds': 908.3200472999997, 'reused_original_queries': 300, 'new_evaluated_queries': 493}. Original300 inference cost was not incurred again. Model loading, integrity reconstruction, hashing and report generation are outside these per-example timings.

Manifests, population hash, exact original config, pre-run checks and original artifact hashes are at the root. `artifact_sources.json` distinguishes reused paths from new paths. New artifacts are in `new_full/`; original300 remain in their original directories. Curves, original images, SLIC, coalitions, embeddings, scores, all phi vectors, positive/matched masks, fit checks, per-example metrics, environment, state hashes, failure evidence, subgroup/full summaries and bootstrap resample means are preserved.

This remains conditional evidence under one frozen B/32 Flickr30k/Telea protocol. Bounding-image ownership does not ensure semantic falsity of a competing caption, whole-region budget overshoot remains, and128-coalition SHAP is approximate. It does not establish universal superiority, causal faithfulness, or generalization to arbitrary settings. Earlier B/16/SugarCrepe results are separate evidence and were not recomputed.

Experiment4R is complete. STOP. No incorrectly ranked query, new seed, larger B/16/SugarCrepe study, new dataset/model/method/grounding analysis or Experiment8 was evaluated. Wait for human review before paper writing.
