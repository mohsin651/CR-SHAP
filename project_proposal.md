# Minimal Research Prototype
## LILI-style Perturbations for SHAP Explanations of CLIP

### Goal

Test whether realistic image perturbations inspired by LILI improve SHAP
explanations of CLIP image-text similarity.

Do NOT build a full research framework yet.
Do NOT implement BERT yet.
Do NOT implement contrastive CR-SHAP yet.
Do NOT train any model.

This is a small proof-of-concept experiment.

---

## Research Question

When SHAP explains CLIP image-text similarity, does replacing standard
image masking with LILI-style LaMa inpainting produce more faithful
explanations?

---

## Hypothesis H1

SHAP using LaMa-inpainted perturbations will produce more faithful
explanations than SHAP using simple mean-color masking.

We will test this using deletion AUC.

Lower deletion AUC = better explanation.

---

## Optional Hypothesis H2

LILI-style mask expansion will improve LaMa-SHAP further because it prevents
the inpainting model from reconstructing information from residual boundaries.

Compare:

1. SHAP + Mean Mask
2. SHAP + LaMa
3. SHAP + LaMa + Mask Expansion

---

# Model

Use pretrained:

    openai/clip-vit-base-patch32

through Hugging Face Transformers.

CLIP must remain completely frozen.

For image I and caption T, explain the CLIP cosine similarity:

    score(I,T) = cosine(image_embedding(I), text_embedding(T))

Do not use a classifier or fine-tune CLIP.

---

# Dataset

Start with Flickr30k.

For the proof of concept, use only 30 correctly matched image-caption pairs.

Use a fixed random seed:

    seed = 42

For every image, use one of its ground-truth captions.

This first experiment is deliberately small.

---

# Image Features

Segment each image using SLIC superpixels.

Use approximately:

    n_segments = 16

Each superpixel is one SHAP feature.

For a coalition:

    1 = keep the superpixel
    0 = remove the superpixel

All methods MUST use exactly the same segmentation and exactly the same
sampled SHAP coalitions for each image.

---

# Method A — Mean-SHAP baseline

For every absent superpixel:

replace its pixels using the global RGB mean of the original image.

Then evaluate:

    CLIP_similarity(perturbed_image, caption)

Use these scores to estimate SHAP values for the superpixels.

---

# Method B — LaMa-SHAP

Use exactly the same SHAP coalitions.

Instead of replacing absent superpixels with mean color:

1. combine absent superpixels into one binary mask;
2. give the original image + mask to pretrained LaMa;
3. use the LaMa output as the perturbed image;
4. calculate CLIP similarity.

Then estimate SHAP values.

This is the main experimental method.

---

# Method C — LaMa-SHAP with LILI-style mask expansion

Same as Method B, except expand/dilate the removal mask before LaMa
inpainting.

Start with:

    dilation radius = 3 pixels

Use a 7x7 square structuring element.

This method tests the mask-expansion idea from LILI.

---

# SHAP Approximation

Use the same 128 sampled coalitions for all three methods for a given image.

Always include:

- empty coalition
- full coalition
- all singleton coalitions
- complements of singleton coalitions

Sample the remaining coalitions randomly.

Use seed 42.

Estimate one SHAP value per superpixel.

Save all raw SHAP values.

---

# Main Evaluation: Deletion Test

For each explanation:

1. Keep only positive SHAP values.
2. Rank superpixels from highest to lowest SHAP value.
3. Starting from the original image, progressively remove the most important
   superpixels.
4. Measure CLIP similarity to the original caption after each removal.
5. Compute area under this deletion curve.

IMPORTANT:

Do NOT evaluate LaMa-SHAP by deleting regions with LaMa.

Use one common independent deletion operator for ALL methods.

Use OpenCV Telea inpainting:

    cv2.INPAINT_TELEA
    radius = 3

This avoids giving LaMa-SHAP an unfair evaluation advantage.

Metric:

    Deletion AUC

Lower = better.

---

# Additional Evaluation: Perturbation Realism

For the generated SHAP perturbations, save example images from:

- Mean masking
- LaMa
- LaMa + expansion

For this first 30-image experiment, visual inspection is enough.

Do NOT implement FID yet.

We mainly want to verify that LaMa perturbations look substantially more
natural and that mask expansion actually removes the intended region.

---

# Outputs

Create:

    results/results.csv

with one row per image and columns:

    image_id
    caption
    mean_shap_deletion_auc
    lama_shap_deletion_auc
    lama_expanded_shap_deletion_auc

Also calculate:

    mean AUC for each method
    median AUC for each method

and paired differences:

    Mean-SHAP - LaMa-SHAP
    Mean-SHAP - LaMa-Expanded-SHAP
    LaMa-SHAP - LaMa-Expanded-SHAP

Remember:

    positive difference means the second method is better
    because lower AUC is better.

Run a paired Wilcoxon signed-rank test:

    Mean-SHAP vs LaMa-SHAP
    Mean-SHAP vs LaMa-Expanded-SHAP

Save:

    results/summary.json

---

# Visualizations

For at least 10 examples save one figure containing:

1. Original image
2. SLIC segmentation
3. Mean-SHAP heatmap
4. LaMa-SHAP heatmap
5. LaMa-expanded-SHAP heatmap
6. Deletion curves for all three methods

Also save examples of actual perturbations generated by each masking method.

---

# Logging

Record runtime separately for:

    Mean-SHAP
    LaMa-SHAP
    LaMa-expanded-SHAP

Record GPU model and peak VRAM if convenient.

---

# Critical implementation checks

Before running all 30 images:

Run only 3 images.

Verify manually that:

1. CLIP similarity for the full coalition equals the original image-caption
   similarity.

2. Mean, LaMa and LaMa-expanded methods use identical SHAP coalitions.

3. The LaMa mask corresponds exactly to the absent SLIC regions.

4. Mask expansion visibly expands the correct mask.

5. LaMa output contains no obvious corrupted dimensions/ranges.

6. SHAP values are not all identical or zero.

7. Deletion ranking actually follows descending positive SHAP values.

Only after these checks pass, run the 30-image experiment.

---

# Decision Rule

This experiment is exploratory and is NOT an ICPR result yet.

After running 30 examples:

PROMISING:

    LaMa or LaMa-expanded has lower mean/median deletion AUC than Mean-SHAP,
    the improvement occurs across many images rather than 2-3 outliers,
    and qualitative explanations look reasonable.

VERY PROMISING:

    relative deletion-AUC improvement >= 5%
    AND paired Wilcoxon p < 0.05
    AND LaMa perturbations visibly reduce masking artifacts.

NOT PROMISING:

    LaMa gives approximately identical or worse deletion AUC,
    despite producing more realistic images.

If the result is NOT PROMISING, do not add more datasets or methods.
Save all results so we can analyze why.

---

# Reproducibility

Use:

    seed = 42

Save:

    configuration
    package versions
    selected image IDs
    captions
    segmentation maps
    sampled coalitions
    raw SHAP values
    deletion curves
    final metrics

Do not overwrite previous experiment results.

---

# Final instruction

The purpose of this implementation is NOT to prove a paper.

It is to answer exactly one question:

    Does LILI-style LaMa perturbation make SHAP explanations
    of CLIP image-text similarity measurably better than ordinary masking?

Keep the implementation modular so that, if the answer is YES, the next
experiment can replace:

    score(I,T)

with:

    score(I,T_positive) - score(I,T_negative)

to test Contrastive Retrieval SHAP.