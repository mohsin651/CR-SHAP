# Experiment 2: Pointwise SHAP versus CR-SHAP

This package adds a separate controlled SugarCrepe experiment. It imports the
validated Experiment 1 functions without modifying or rerunning Experiment 1.
Use the existing `.venv` and dependencies from the workspace root.

```powershell
.venv\Scripts\python.exe -m experiment2.data
.venv\Scripts\python.exe -m experiment2.screen
.venv\Scripts\python.exe -m pytest tests/test_experiment2.py -q
.venv\Scripts\python.exe -m experiment2.run --stage pilot
# Inspect the five pilot figures and verify checks.json before the full run:
.venv\Scripts\python.exe -m experiment2.run --stage full
```

The data command downloads the seven official category JSONs from a pinned
SugarCrepe Git commit and COCO val2017 from the official image S3 bucket.
`source.json` records URLs, revision and SHA256 hashes. No renamed category
mapping is introduced. Screening scores every official instance, caches one
normalized embedding per unique working image and caption, and reports all
correct/incorrect counts. Accuracy uses the validated max-side-512 preprocessing.

Selection uses seed 42, balanced water filling across correctly ranked category
pools, and uniform sampling without replacement. A seeded category permutation
chooses which categories receive extra slots. Exact IDs and captions are saved
before explanation computation. The five pilot instances are the first five in
category round-robin order. No sample replacement or parameter optimization is
permitted. Fresh output paths are required; existing directories are refused.

For each coalition, a single mean-masked image is encoded once and compared with
both cached caption embeddings. Three regressions estimate positive, negative,
and direct contrastive values. The negative regression serves the required
linearity check, not an additional explanatory method. A mismatch exceeding
1e-10 or any sanity-check failure stops execution. Full runs require a passed
five-instance pilot with matching configuration and selected IDs.

## User-approved deletion convention

Only strictly positive SHAP regions are deleted, descending by value with
ascending-index ties. The validated Telea implementation is reused exactly.
Preference is sigmoid(pretrained gamma × raw cosine margin); SHAP uses raw
cosines, never scaled logits or preference probabilities.

After positive regions are exhausted, hold the final preference constant to x=1
solely for integration. A zero-positive case remains constant from the original
state and is flagged. Do not delete zero/negative regions. Actual-state curves
remain distinct from AUC tails. Unreached 25%/50% deletion states are missing;
report reached-case rates, coverage, observed flips over all instances, and a
paired comparison restricted to cases reached by both methods. The full-mask
global-mean fallback for Telea is unchanged from Experiment 1.

## Outputs and interpretation

`results/experiment_2` contains per-instance and category CSVs, summary JSON,
raw scores/SHAP/masks/segmentations, deletion evidence, hashes, checks, environment,
selected IDs, fifteen outcome-stratified figures, and the exploratory margin
scatter plot. Masks are reconstructible exactly from saved coalitions and SLIC
maps; pixel-mask hashes and perturbed-image hashes are saved as well.

Paired Wilcoxon is two-sided; paired percentile bootstrap uses 10,000 resamples
with seed 42. No significance tests are performed on the small category strata.
Figures use five largest improvements, five closest-to-zero remaining cases, and
five largest worsenings, with deterministic ID tie breaks. Selection IDs and
the rule are saved.

Timing records the shared coalition generation/scoring cost separately. Each
method's runtime is shared cost plus its own regression; the common inference
cost is incurred once in the actual experiment.

After inspecting the unchanged fixed results, choose the research classification
using the supplied request's criteria and generate the self-contained report:

```powershell
.venv\Scripts\python.exe -m experiment2.report --classification MIXED --rationale 'State the observed global and category evidence here.'
```

Classification is scientific interpretation, not an automatically tuned rule.
VERY PROMISING requires ≥5% mean improvement, paired p<.05, a positive 95% mean
difference CI, a majority of CR wins, and improvement in at least three categories.
The remaining classes require judging practical consistency, category signals,
and RankFlip effects; a single seed cannot demonstrate reproducibility of those
signals. Reports must distinguish observations from replication claims.
