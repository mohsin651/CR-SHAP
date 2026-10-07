# LILI-style LaMa perturbations for SHAP explanations of CLIP

## Question and setup

Does replacing mean-color image masking with realistic LaMa inpainting improve
SHAP explanations of CLIP image-text cosine similarity? A second hypothesis
tests whether expanding removal masks improves explanations further.

This exploratory experiment used 30 randomly selected Flickr30k images, each
paired with one ground-truth caption, and seed 42. Pairs were annotation-matched,
not filtered for successful CLIP retrieval. The pretrained Hugging Face model
`openai/clip-vit-base-patch32` remained frozen; no model was trained.

Images retained their aspect ratio, with a maximum side of 512 pixels. SLIC
requested approximately 16 superpixels, using compactness 30 to avoid excessive
region merging. Constrained Kernel SHAP regression estimated region contributions
from 128 identical coalitions per image across three methods. Coalitions included
empty/full sets, every singleton, every singleton complement, and random remaining
sets. The methods were:

1. **Mean-SHAP:** replace absent regions with the original image's global RGB mean.
2. **LaMa-SHAP:** inpaint absent regions using pretrained big-LaMa.
3. **Expanded LaMa-SHAP:** dilate removal masks by 3 pixels using a 7×7 square,
   then apply the same LaMa model.

## Evaluation and results

All explanations were evaluated with the same independent OpenCV Telea deletion
operator, radius 3. Only positive SHAP regions were deleted, in descending order.
Deletion AUC used raw CLIP cosine similarity against the fraction of pixels
removed; **lower is better**. Once all positive regions were removed, the last
score was held constant to 100% deletion so all curves shared one integration
domain. A fully masked image used a global-mean fallback because Telea has no
remaining context.

| Method | Mean AUC | Median AUC | Wins against Mean-SHAP | Paired Wilcoxon p | Mean explanation time |
|---|---:|---:|---:|---:|---:|
| Mean-SHAP | 0.239093 | 0.235756 | — | — | 0.80 s |
| LaMa-SHAP | 0.236683 | 0.234268 | 16/30 | 0.6554 | 5.74 s |
| Expanded LaMa-SHAP | 0.241272 | 0.241531 | 13/30 | 0.4771 | 5.78 s |

Mean paired AUC differences (first minus second; positive favors the second):

- Mean-SHAP minus LaMa-SHAP: **+0.002411**.
- Mean-SHAP minus Expanded LaMa-SHAP: **−0.002179**.
- LaMa-SHAP minus Expanded LaMa-SHAP: **−0.004589**.

LaMa improved mean AUC by **1.01%**; expansion worsened it by **0.91%** relative
to Mean-SHAP. Wilcoxon tests were two-sided and uncorrected for multiple comparisons.
LaMa explanation generation was approximately seven times slower than mean masking.

## Validation and limitations

A three-image pilot was visually reviewed before the final run. Five unit tests
passed. All 30 final examples passed checks for full-coalition/original-score
agreement, shared coalitions, mask correctness, expansion, valid image outputs,
nontrivial SHAP values, SHAP efficiency, and descending positive deletion rankings.
Identical LaMa empty masks also produced matching scores. GPU TF32 arithmetic and
TorchScript graph optimization were disabled after reproducibility checks exposed
batch-size and first-call numerical differences. Earlier runs were preserved and
excluded from conclusions.

Visual inspection found that LaMa reduces some flat-fill artifacts, but can
reconstruct or distort partially masked objects. Realistic-looking perturbations
therefore do not guarantee successful feature removal or faithful attribution.

The experiment covers one model, 30 images, one seed, and an approximate SHAP
estimator. The positive-only deletion curve's constant tail can affect comparisons
when methods assign different amounts of image area positive importance. Telea
provides a common evaluator but does not establish causal faithfulness. No FID,
additional datasets, BERT, or contrastive CR-SHAP were implemented.

## Interpretation and saved evidence

**No convincing support for either hypothesis in this run.** LaMa's small mean
improvement was not statistically significant, and mask expansion worsened mean
and median AUC. Under the prototype decision rule, this is **not promising yet**;
it does not prove that LaMa cannot help in other settings. The implementation did
not expand to additional datasets or methods.

The validated run is `results/full_002`, following `results/pilot_004`. Metrics are
in `results/results.csv` and `results/summary.json`. Configuration, package versions,
selected pairs, segmentation maps, shared coalitions, raw SHAP values, model scores,
deletion curves, timing, and ten explanation figures are saved in the run directory.

For discussion with ChatGPT: assess whether the small effect reflects ineffective
feature removal, the SHAP approximation, or the deletion metric. Separate these
possible explanations from demonstrated findings; prioritize analysis of the saved
results before proposing a larger experiment.
