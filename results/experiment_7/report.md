# Experiment 7: final CLIP ViT-B/16 backbone replication

Both datasets replicate the positive CR-SHAP direction with paired confidence intervals above zero. The advantage is not specific to ViT-B/32 in these two settings.

## Scientific question and frozen design

Does Pointwise-to-CR SHAP improvement persist under a second CLIP backbone on controlled SugarCrepe pairs and natural Flickr30k retrieval? The arms are analyzed separately and never pooled. Each explains exactly 200 examples from a fresh B/16-eligible pool. Experiments 1–6 were not rerun, modified, or optimized.

Model: `openai/clip-vit-base-patch16`, pinned revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`. All parameters are frozen, eval mode, with no gradients/optimizer/training. Full state hashes before/after screening and each pilot/full explanation run agree. Package, torch/CUDA and GPU details are saved per stage; weights and processor are pinned to the same revision.

The unchanged working-image preparation is EXIF transpose, RGB, aspect-preserving max-side512 followed by CLIP processor resize/center crop. B/16 has 224×224 input, 14×14 transformer patches, and 197 tokens including CLS. These are not the SHAP features: SLIC still requests16 regions, compactness30, start_label0, channel_axis−1, scikit-image0.24.0 defaults. No methodological parameter needed to change for the patch geometry.

Both arms use128 unique coalitions, seed42, mandatory empty/full/singletons/complements, the unchanged constrained Kernel SHAP solver, and original global-RGB mean fill. One normalized image embedding per coalition supplies both caption cosines. Raw positive cosine is Pointwise; positive-minus-negative raw cosine is CR. Negative-caption regression validates Shapley linearity. Coalitions, embeddings, scores, all phi vectors, masks/hashes and fit diagnostics are preserved.

Primary: MatchedArea-PDAUC-50, signed ranking over all SLIC regions, index ties ascending, whole-region first crossings of10/20/30/40/50%, actual pixel fractions, Telea radius3 on the original image, unchanged full-mask mean fallback, original-state inclusion, normalized six-state trapezoidal area, no extrapolation. Preference uses sigmoid(gamma × margin) with each model’s pretrained logit scale. PositiveOnly-PDAUC remains secondary, deleting strictly positive regions and extending the terminal preference to100% for integration only.

## Screening and sample construction

Selection seed2027 was fixed before outcomes. Each arm uniformly permutes eligible unique image IDs and uniformly chooses one eligible instance per image. This meets the200 unique-example target while avoiding repeated-image dependence. SugarCrepe is not category-balanced: it samples images from the complete B/16-eligible pool rather than replicating the old equal-category weighting. This sampling difference limits direct effect-size comparisons. All initial200 segmentations are checked before explanations; infeasible examples are replaced only from the frozen reserve order, never from outcomes.

### sugarcrepe

Complete B/16 screening: 7511 instances, 1560 unique images; 5768 strictly correct, 1743 incorrect, 0 ties. Selection:200 unique images, 1378 eligible images, 0 technical exclusions. Gamma=100.000000.

Positive/negative captions are the existing official SugarCrepe pair. All7511 official instances were freshly screened using B/16; B/32 correctness was not inherited.

Selected category counts: {'add_att': 13, 'add_obj': 48, 'replace_att': 15, 'replace_obj': 39, 'replace_rel': 20, 'swap_att': 41, 'swap_obj': 24}.

### flickr30k

Complete B/16 screening: 1000 instances, 1000 unique images; 824 strictly correct, 176 incorrect, 0 ties. Selection:200 unique images, 824 eligible images, 0 technical exclusions. Gamma=100.000000.

The standard Karpathy test set retains1000 images and5000 owned captions. New normalized B/16 embeddings and the full similarity matrix determine the highest-scoring ground-truth T+ and strongest other-image T− independently for every image. B/32 caption competitors/embeddings are not reused.

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| image_to_text | 82.40% | 96.70% | 99.00% |
| text_to_image | 62.20% | 85.58% | 91.94% |

Original-margin distribution: {'mean': 0.027423048824071886, 'median': 0.024288266897201538, 'standard_deviation': 0.03025317386803163, 'minimum': -0.05431187152862549, 'maximum': 0.13920345902442932, 'std_ddof': 0}.

Retrieval is an original-model sanity check, not an improvement caused by post-hoc SHAP.

## Headline primary results

| Dataset | Pointwise mean / median | CR mean / median | Mean / median difference | Relative improvement | CR / Point wins / ties | Paired95% CI | Wilcoxon statistic / p |
|---|---|---|---|---:|---|---|---|
| sugarcrepe | 0.682151 / 0.689871 | 0.630996 / 0.622961 | +0.051155 / +0.028554 | 7.50% | 135 / 64 / 1 | [+0.036980, +0.065592] | 4943 / 7.48953e-10 |
| flickr30k | 0.724215 / 0.806475 | 0.662514 / 0.712867 | +0.061701 / +0.027615 | 8.52% | 145 / 54 / 1 | [+0.044316, +0.080363] | 4639 / 6.6162e-11 |

n=200 paired unique images in each arm. Positive AUC/preference differences mean Pointwise minus CR; positive margin-drop differences mean CR minus Pointwise. Tests are paired two-sided Wilcoxon and paired percentile bootstrap10000 resamples, seed42. No cross-dataset pooled p-value or cross-backbone effect-size significance test was performed. Secondary budgets are unadjusted.

## Secondary PositiveOnly-PDAUC

| Dataset | Point mean / median | CR mean / median | Mean / median difference | Relative improvement | CR / Point wins / ties | 95% CI | Wilcoxon p |
|---|---|---|---|---:|---|---|---:|
| sugarcrepe | 0.626402 / 0.615305 | 0.570984 / 0.553544 | +0.055418 / +0.036387 | 8.85% | 134 / 66 / 0 | [+0.039329, +0.072036] | 1.02331e-09 |
| flickr30k | 0.621370 / 0.640652 | 0.571518 / 0.589500 | +0.049852 / +0.038488 | 8.02% | 125 / 75 / 0 | [+0.027595, +0.071620] | 1.13898e-05 |

## Secondary fixed budgets

| Dataset | Budget | Actual area Point / CR | Preference Point / CR | Margin drop Point / CR | RankFlip Point / CR | Neither / Point only / CR only / Both | Exact McNemar p |
|---|---:|---|---|---|---|---|---:|
| sugarcrepe | 10% | 15.39% / 15.18% | 0.711169 / 0.673058 | 0.006744 / 0.009272 | 18.00% / 19.50% | 151 / 10 / 13 / 26 | 0.677639 |
| sugarcrepe | 20% | 25.08% / 24.44% | 0.683470 / 0.624987 | 0.008323 / 0.012538 | 24.00% / 31.00% | 127 / 11 / 25 / 37 | 0.0288167 |
| sugarcrepe | 30% | 34.63% / 34.34% | 0.655201 / 0.590817 | 0.010501 / 0.014574 | 24.50% / 35.00% | 120 / 10 / 31 / 39 | 0.00145049 |
| sugarcrepe | 40% | 44.40% / 44.30% | 0.629680 / 0.556485 | 0.012045 / 0.016782 | 25.50% / 42.50% | 109 / 6 / 40 / 45 | 3.1028e-07 |
| sugarcrepe | 50% | 54.15% / 54.10% | 0.599946 / 0.526281 | 0.013863 / 0.018562 | 31.00% / 45.50% | 93 / 16 / 45 / 46 | 0.000264279 |
| flickr30k | 10% | 15.31% / 14.85% | 0.776330 / 0.747427 | 0.012752 / 0.016322 | 16.00% / 20.00% | 152 / 8 / 16 / 24 | 0.15159 |
| flickr30k | 20% | 25.14% / 24.44% | 0.711648 / 0.654513 | 0.017701 / 0.024529 | 26.00% / 32.00% | 127 / 9 / 21 / 43 | 0.0427739 |
| flickr30k | 30% | 34.62% / 34.41% | 0.671567 / 0.601549 | 0.021743 / 0.029153 | 31.00% / 36.00% | 113 / 15 / 25 / 47 | 0.15386 |
| flickr30k | 40% | 44.31% / 44.47% | 0.641509 / 0.536742 | 0.024523 / 0.034319 | 32.00% / 45.50% | 98 / 11 / 38 / 53 | 0.000141971 |
| flickr30k | 50% | 54.26% / 54.48% | 0.599060 / 0.481152 | 0.027401 / 0.038565 | 39.50% / 48.00% | 91 / 13 / 30 / 66 | 0.0137182 |

All paired preference/margin-drop means, medians, differences, wins, Wilcoxon results and CIs are in each arm’s `summary.json`. Exact McNemar is the two-sided binomial test on discordant paired flips.

## Descriptive cross-backbone comparison

| Dataset | Backbone | n | Pointwise matched PDAUC | CR matched PDAUC | Relative improvement | Paired mean-difference95% CI |
|---|---|---:|---:|---:|---:|---|
| sugarcrepe | ViT-B/32 | 699 | 0.670330 | 0.625471 | 6.69% | [+0.036384, +0.053332] |
| sugarcrepe | ViT-B/16 | 200 | 0.682151 | 0.630996 | 7.50% | [+0.036980, +0.065592] |
| flickr30k | ViT-B/32 | 300 | 0.681292 | 0.618664 | 9.19% | [+0.043120, +0.081947] |
| flickr30k | ViT-B/16 | 200 | 0.724215 | 0.662514 | 8.52% | [+0.044316, +0.080363] |

B/32 values are read verbatim from saved summaries; no B/32 inference or analysis was rerun. SugarCrepe B/32 uses the Experiment3 matched-area robustness result and its image-cluster CI. The table is descriptive: model-dependent eligibility, negative choices, sample size and category/image weighting differ.

## Caption and correctness overlap diagnostics

Across all1000 common Flickr30k test images: {'all1000_same_positive_fraction': 0.57, 'all1000_same_negative_fraction': 0.256, 'all1000_both_same_fraction': 0.148, 'both_correct': 726, 'b16_only_correct': 98, 'b32_only_correct': 67, 'neither_correct': 109}

Caption overlap is defined by saved caption IDs over the same ownership pool. These are descriptive diagnostics, not a new hypothesis or explanation-selection criterion.

## Technical validation and runtime

sugarcrepe: a five-image technical pilot was independently verified before either full arm. All200 examples passed pinned B/16/frozen checks, screening-score/caption-vector reproduction, unchanged segmentation,128 unique shared coalitions, constrained regression/linearity, deterministic first-crossing masks and unchanged evaluator formulas. Maximum Shapley linearity error 1.67e-16. Full state hashes match before/after. Independent reconstruction verified25,600 coalition masks and perturbed images plus embeddings, regressions, deletion rankings, budget masks and paired statistics. Protected prior files unchanged: 13,710.

Runtime seconds: {'screening_seconds': 38.006850899999336, 'shared_coalition_inference_seconds': 369.21854449999955, 'point_regression_seconds': 0.06384179999986372, 'cr_regression_seconds': 0.028445100000681123, 'shared_evaluation_seconds': 400.4833631000056}.

flickr30k: a five-image technical pilot was independently verified before either full arm. All200 examples passed pinned B/16/frozen checks, screening-score/caption-vector reproduction, unchanged segmentation,128 unique shared coalitions, constrained regression/linearity, deterministic first-crossing masks and unchanged evaluator formulas. Maximum Shapley linearity error 2.08e-16. Full state hashes match before/after. Independent reconstruction verified25,600 coalition masks and perturbed images plus embeddings, regressions, deletion rankings, budget masks and paired statistics. Protected prior files unchanged: 13,710.

Runtime seconds: {'screening_seconds': 14.92107279999982, 'shared_coalition_inference_seconds': 346.4512893000192, 'point_regression_seconds': 0.054622099985863315, 'cr_regression_seconds': 0.02269259999775386, 'shared_evaluation_seconds': 326.9114066000093}.

Model loading, snapshot hashing, disk serialization and report generation are outside the recorded inference/evaluator costs. Both targets share coalition encoding and identical evaluator states are cached. No old-model embeddings were reused.

## Final answers and limitations

**Question1 — Does CR-SHAP outperform Pointwise under B/16 on sugarcrepe?** Positive mean improvement with CI above zero. Relative effect +7.50%, Wilcoxon p=7.48953e-10, CI [+0.036980, +0.065592].

**Question2 — Does CR-SHAP outperform Pointwise under B/16 on flickr30k?** Positive mean improvement with CI above zero. Relative effect +8.52%, Wilcoxon p=6.6162e-11, CI [+0.044316, +0.080363].

**Question3 — Is the direction consistent across backbones?** Both datasets replicate the positive CR-SHAP direction with paired confidence intervals above zero. The advantage is not specific to ViT-B/32 in these two settings.

**Question4 — Is there evidence that the advantage is specific to B/32?** The positive effects in both B/16 arms argue against strict B/32 specificity in these settings; they do not establish generality across arbitrary models.

This fixed-scope replication uses one B/16 checkpoint, one selection seed and200 correctly ranked images per dataset. SugarCrepe image-uniform sampling differs from the older category-balanced confirmation, and Flickr30k constructs model-specific caption competitors. Approximate Kernel SHAP, SLIC granularity, mean masking, positive-only tail conventions, whole-region overshoot, and Telea perturbations constrain interpretation. There is no training, objective tuning, effect-size matching, extra baseline, grounding evaluation, dataset expansion or post-hoc robustness run.

## Artifacts

Fresh full screening/eligible pools, new caption winners and model-specific embeddings are in `data/experiment7_screening/{sugarcrepe,flickr30k}`. Selected examples, pre-generated reserves, preflight masks and technical exclusions were frozen before explanations. Each `results/experiment_7/<arm>/full` preserves configs, package/model provenance, state/source/input hashes, original images, SLIC, shared coalitions, normalized embeddings, both cosine caches, three SHAP vectors, masks, raw deletion curves, per-example metrics, statistics, timings and independent verification.

Experiment7 is complete. STOP ALL EXPERIMENTATION. No larger sample, additional seed, model, dataset, baseline, grounding experiment or Experiment8 was launched. Human review decides whether to write the paper.
