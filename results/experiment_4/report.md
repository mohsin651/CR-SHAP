# Experiment 4: external validation of CR-SHAP on natural Flickr30k retrieval

The primary matched-area endpoint supports a CR-SHAP advantage in this sample.

## Dataset, retrieval, and frozen sample

Original Karpathy TEST annotations: 1,000 unique images, 5,000 captions, exactly five per image. Image-caption ownership and an independent retrieval annotation mirror were validated. Captions are used verbatim from the original annotations.

Full retrieval screening: 793 strictly correctly ranked images (79.30%), 207 incorrect or tied, 0 exact margin ties.

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| image_to_text | 79.30% | 95.00% | 98.10% |
| text_to_image | 58.82% | 83.44% | 90.08% |

Retrieval metrics describe the original zero-shot CLIP model. Post-hoc CR-SHAP does not change retrieval accuracy.

| Screening quantity | Mean | Median | Std (population) | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| original_positive_cosine | 0.340400 | 0.341085 | 0.028872 | 0.253625 | 0.448205 |
| original_negative_cosine | 0.316960 | 0.316420 | 0.021771 | 0.240640 | 0.394556 |
| original_margin | 0.023441 | 0.022533 | 0.028671 | -0.064924 | 0.126378 |

Uniform seed-2026 sampling selected 300 unique correctly ranked query images. All initial images were checked for coalition feasibility before explanation outcomes; 0 technical exclusions and 0 reserve checks are documented in `technical_exclusions.json`. Replacements followed the pre-generated seeded reserve list.

T+ is the highest-scoring ground-truth caption. T- is the highest-scoring caption owned by another test image, selected from the complete 5,000-caption pool. Caption IDs, ownership, scores, and all screening rows are saved in `data/flickr30k_screening_exp4/screening.csv`.

Frozen CLIP `openai/clip-vit-base-patch32`, commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; eval mode, no gradients. EXIF transpose, RGB, aspect-preserving maximum side 512, then the unchanged CLIP processor/tokenizer. SLIC: 16 requested regions, compactness 30, start label 0, channel axis -1, scikit-image 0.24.0 defaults. Kernel SHAP: unchanged constrained solver, 128 unique shared coalitions, seed 42, global RGB mean masking. One normalized image embedding per coalition supplies both raw cosine scores.

## Paired endpoints

| Endpoint | Point mean / median | CR mean / median | Mean / median difference | Relative improvement | CR / Point wins / ties | Wilcoxon statistic / p | Paired 95% CI |
|---|---|---|---|---:|---|---|---|
| Primary: MatchedArea-PDAUC-50 | 0.681292 / 0.750801 | 0.618664 / 0.651444 | +0.062628 / +0.030781 | 9.19% | 193 / 107 / 0 | 13475 / 1.43468e-09 | [+0.043120, +0.081947] |
| Secondary: PositiveOnly-PDAUC | 0.587933 / 0.613426 | 0.514820 / 0.510955 | +0.073113 / +0.055750 | 12.44% | 200 / 100 / 0 | 13005 / 1.96448e-10 | [+0.051030, +0.095774] |

n=300. Paired two-sided Wilcoxon tests and 10,000 paired percentile bootstrap resamples, seed 42, over unique query images. No image-cluster correction is needed for repeated query images because each occurs once. Positive AUC/preference differences mean Pointwise minus CR; positive margin-drop and rank-flip differences mean CR minus Pointwise.

Both evaluations use Telea radius 3 on the original image with each cumulative whole-region mask; the validated full-mask global-mean fallback is unchanged. Primary ranking uses all signed SHAP values descending, index ties ascending. Each 10/20/30/40/50% budget uses the first whole-region crossing. AUC integrates the original state and five crossing states against actual areas and divides by the final area; repeated crossings have zero-width intervals, with no tail. Secondary deletion removes strictly positive regions only, extending the final preference constantly to 100% solely for integration. Preference is sigmoid(gamma times cosine margin), gamma approximately 100.

## Secondary fixed budgets

| Budget | Actual area Point / CR | Preference Point / CR | Margin drop Point / CR | Flip Point / CR | Neither / Point only / CR only / Both | Exact McNemar p |
|---|---|---|---|---|---|---:|
| 10% | 15.67% / 15.15% | 0.714802 / 0.681972 | 0.012825 / 0.017035 | 22.67% / 29.00% | 198 / 15 / 34 / 53 | 0.00939924 |
| 20% | 24.79% / 24.75% | 0.676075 / 0.607422 | 0.016783 / 0.023994 | 28.00% / 37.67% | 171 / 16 / 45 / 68 | 0.000264279 |
| 30% | 34.64% / 34.64% | 0.628664 / 0.549038 | 0.021329 / 0.028691 | 35.33% / 44.67% | 145 / 21 / 49 / 85 | 0.00109324 |
| 40% | 44.39% / 43.99% | 0.601392 / 0.495695 | 0.024881 / 0.032996 | 35.33% / 49.33% | 135 / 17 / 59 / 89 | 1.39687e-06 |
| 50% | 54.50% / 54.23% | 0.562834 / 0.457053 | 0.027830 / 0.036323 | 41.67% / 54.00% | 112 / 26 / 63 / 99 | 0.000109926 |

Exact two-sided McNemar uses a binomial test on discordant pairs. Budget tests are secondary and unadjusted. Complete preference/margin-drop statistics and CIs are saved in `summary.json`.

## Exploratory difficulty relationships

- original_margin versus matched PDAUC difference: Spearman rho=-0.06239935999288881, p=0.2813342708200413.
- original_negative_cosine versus matched PDAUC difference: Spearman rho=0.019493994377715304, p=0.7366685654434734.

These correlations are exploratory and do not redefine the primary hypothesis.

## Evidence and interpretation

Independent verification reconstructed all 1,000 retrieval winners, recall metrics, reserve sampling, 38,400 coalition masks/perturbations, shared cosine scores, constrained regressions, signed rankings, budget masks, formulas, and paired statistics. Maximum linearity error: 2.36e-16. All 7,477 protected prior files remained unchanged. SugarCrepe was not rerun.

This measures external validity under the frozen CLIP/Telea evaluator. Other-image ownership does not ensure that a retrieved caption is semantically false: natural caption overlap is possible. Conditioning on correctly ranked images, technical exclusions, approximate SHAP, whole-region overshoot, shared candidate captions, one model and sampling seed limit generalization. This does not establish causal faithfulness. The primary endpoint stays matched-area regardless of the secondary result.

Per-example original images, segmentations, coalitions, normalized image/text embeddings, both cosine caches, all three SHAP vectors, hashes, deletion masks and curves are preserved in `example_000` through `example_299`. Timing separates shared coalition inference, method regressions, and evaluator cost. Fifteen outcome-stratified figures are diagnostic; selection is deterministic and not representative.

![improvement: image 6317293855](figures/improvement_126.png)

![improvement: image 4846324908](figures/improvement_148.png)

![improvement: image 2844641033](figures/improvement_128.png)

![improvement: image 3903017514](figures/improvement_237.png)

![improvement: image 1395410911](figures/improvement_050.png)

![near_zero: image 5501939468](figures/near_zero_097.png)

![near_zero: image 3532476966](figures/near_zero_224.png)

![near_zero: image 2549933281](figures/near_zero_021.png)

![near_zero: image 3155400369](figures/near_zero_127.png)

![near_zero: image 4727540499](figures/near_zero_057.png)

![worsening: image 7329031116](figures/worsening_275.png)

![worsening: image 3039200576](figures/worsening_041.png)

![worsening: image 3612485097](figures/worsening_265.png)

![worsening: image 3421480658](figures/worsening_259.png)

![worsening: image 1404832008](figures/worsening_015.png)
