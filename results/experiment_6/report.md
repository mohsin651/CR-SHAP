# Experiment 6: Flickr30k Entities grounding of frozen SHAP explanations

## Scientific question

Do the explanations that better explain relative CLIP retrieval preference also localize human-annotated visual entities in T+? This experiment measures grounding, not deletion faithfulness, using only Pointwise SHAP and CR-SHAP. No model inference or SHAP recomputation occurred.

## Frozen annotation sample and targets

Original frozen sample: **300**. Successfully caption-matched: **300**. Grounding-evaluable: **299**. Excluded for annotation/technical reasons: **1**. All excluded IDs and reasons are in `exclusions.json`; no replacement or outcome-based filtering occurred.

Annotations come from the [official Flickr30k Entities repository](https://github.com/BryanPlummer/flickr30k_entities), pinned at commit `68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b`. Its original sentence/XML parser is archived with the annotation files. The original Karpathy caption ID determines the sentence index; tokens must match exactly after HTML unescape, case folding, and punctuation whitespace normalization. Words and punctuation are not semantically substituted. Original caption text, IDs, index and entity sentence are retained.

For each T+, all explicit valid boxes of all linked visual phrases are included. Multiple boxes per phrase are retained, duplicate boxes are deduplicated, and overlaps contribute once in a union mask. Nonvisual/id0 phrases and phrases without explicit boxes contribute no target. Scene-type phrases with explicit boxes are included; scene/no-box flags alone do not create a whole-image box. Invalid geometry causes a documented technical exclusion. Targets and rules were saved before attributions were scored.

## Coordinates and attribution representation

Official parser coordinates are zero-based inclusive. Maxima are increased by one to obtain half-open pixel edges, clipped to the raw image dimensions, transformed through any EXIF orientation, and scaled by the exact saved working/oriented dimension ratios. Pixel-center inclusion uses ceil(edge − 0.5) raster bounds. No CLIP center crop is applied to ground truth: both boxes and SLIC are evaluated on the saved working image. Original/transformed boxes, dimensions, clipping and orientation are recorded per example.

Nonidentity EXIF cases: 0; raw-to-working resized cases: 0. A fixed technical pilot checked image identity, box alignment, SLIC, conserved energy, pointing and area; overlays were visually reviewed before full statistical analysis.

Saved SHAP values are contributions of whole SLIC regions. The positive spatial density for a pixel in region j is max(phi_j,0)/n_j, where n_j is its pixel count. This preserves each region’s total positive contribution and prevents region area from accidentally multiplying energy. Both metrics use this same map for both methods. EIB numerator is density summed inside the GT union; denominator is total positive SHAP mass. Numerator, denominator and ratio are saved.

Pointing Game chooses the greatest positive density region, with the lowest region index for equal densities, then the pixel nearest that region’s centroid, with row-major y/x ties. The location is always within the selected region. This is a region-aware tie rule for the plateau, rather than an arbitrary first boundary pixel. It ranks contribution density, not the total-contribution ranking used for prior deletion experiments. Zero-positive-energy cases remain in the sample as Pointing failures and EIB=0, with explicit flags.

## Headline paired results

| Metric | Pointwise SHAP | CR-SHAP | CR − Pointwise | Paired 95% CI | p |
|---|---:|---:|---:|---|---:|
| Primary: Pointing Game ↑ | 83.61% | 78.60% | -5.02 pp | [-9.3645, -0.6689] pp | 0.0356978 |
| Secondary: EIB ↑ | 0.698371 | 0.696871 | -0.001501 | [-0.0158, +0.0132] | 0.661158 |
| Context: positive attribution area ↓ | 81.12% | 60.26% | -20.86 pp | [-22.7558, -19.0112] pp | 9.2424e-46 |

Paired n=299. All difference CIs use 10,000 paired percentile bootstrap resamples over unique images, seed 42. Exact two-sided McNemar on discordant pointing outcomes is the primary binary test; secondary EIB uses paired two-sided Wilcoxon. Tests are unadjusted. The area p-value describes selectivity only.

Pointing paired counts: {'neither': 34, 'point_only': 30, 'cr_only': 15, 'both': 220}. Mean GT union coverage: 60.88%; broad unions can make localization permissive.

EIB medians: Pointwise 0.728928, CR 0.763068; median paired difference +0.000909; CR wins 154, Pointwise wins 142, ties 3; Wilcoxon statistic 21332.

Positive area medians: Pointwise 80.73%, CR 60.95%; median paired difference -18.39 pp. Zero-positive-energy cases: Pointwise 0, CR 0.

## Conclusions

**1. Localization:** Pointing Game is lower for CR-SHAP (exact McNemar p=0.0356978); EIB is uncertain (Wilcoxon p=0.661158). A nonsignificant difference does not establish equivalence.

**2. Selectivity:** CR-SHAP uses less positive image area: 81.12% versus 60.26%, paired difference -20.86 percentage points, CI [-22.7558, -19.0112]. Area is contextual, not grounding quality.

**3. Faithfulness versus grounding:** The grounding results indicate a tradeoff on at least one spatial metric, despite the previously observed retrieval-deletion advantage.

Experiments 3–5 tested how perturbations change CLIP retrieval preference. Human box overlap measures a distinct property; improved deletion AUC is not proof of better entity grounding. Bounding-box overlap does not establish causal localization or human interpretability. Contrastive evidence can lie outside the positive-caption entities, but these results alone do not explain why.

## Validation, limitations and reproducibility

All 299 evaluable cases passed independent saved-evidence checks: exact reused SHAP/SLIC, GT raster, conserved density, continuous-box point membership, separate region-overlap energy formula, positive area and paired statistics. All 11,860 protected prior files remained unchanged. Unit tests cover token identity, scaling/inclusive endpoints, clipping, EXIF orientation, region energy conservation, centroid ties and zero-energy handling.

The existing 300 correctly ranked retrieval examples are not the whole Flickr30k population. Union boxes may include large backgrounds/scene entities and do not provide pixel-accurate segmentation. Coarse SLIC plateaus require a deterministic representative point; conserved density changes the peak from the greatest whole-region contribution in some examples. Positive-only mass ignores negative evidence. One sample/model/segmentation, annotation availability, and shared candidate captions limit generalization. No new aggregation, box subset or normalization was chosen using outcomes.

Artifacts: matched/excluded manifests, frozen config/source hashes, original/transformed phrase boxes and union masks in `targets/`, density maps and reused phi/SLIC in `examples/`, per-example results, statistics, environment, geometry pilot/review, verification and deterministic figures. Dataset/parser hashes and references to each reused Experiment 4 artifact are retained.

## Diagnostic examples

For each pointing outcome stratum, choose the first two image IDs lexicographically. Additionally show the three largest EIB worsenings for CR among examples with positive CR energy outside the GT union, ties by image ID. These illustrate possible outside-box discriminative evidence, without establishing its semantics or explaining the retrieval mechanism. Outcome-based figure selection is diagnostic, not quantitative evidence.

![both_grounded: 10287332](figures/both_grounded_063.png)

![both_grounded: 1082250005](figures/both_grounded_264.png)

![cr_only_grounded: 2391094555](figures/cr_only_grounded_295.png)

![cr_only_grounded: 246231741](figures/cr_only_grounded_161.png)

![point_only_grounded: 102617084](figures/point_only_grounded_149.png)

![point_only_grounded: 1255504166](figures/point_only_grounded_150.png)

![neither_grounded: 1313869424](figures/neither_grounded_191.png)

![neither_grounded: 139245992](figures/neither_grounded_086.png)

![outside_box_evidence: 1255504166](figures/outside_box_evidence_150.png)

![outside_box_evidence: 3278581900](figures/outside_box_evidence_207.png)

![outside_box_evidence: 4846324908](figures/outside_box_evidence_148.png)

Experiment 6 is complete. No Grad-ECLIP grounding, further dataset, model, or Experiment 7 was run. Stopped for human review.
