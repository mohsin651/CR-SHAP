# LILI-style SHAP explanations of frozen CLIP

Small implementation of `project_proposal.md`: 16 SLIC regions, 128 shared
coalitions, global-mean / pretrained LaMa / LaMa with a 7×7 mask dilation,
and common OpenCV Telea deletion. No training or contrastive extension.

SLIC requests 16 regions with compactness 30 to avoid aggressive merging on
natural photographs. Actual region counts are recorded; all methods share them.

## Setup (PowerShell)

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
# For an NVIDIA GPU, replace the CPU torch wheel:
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pytest -q
```

`requirements-lock.txt` records the installed environment, including the CUDA
wheel used for verification. Use the same PyTorch index when installing that
lock file on an NVIDIA machine.

## Download and run

```powershell
.venv\Scripts\python.exe -m lili_shap.data --destination data/flickr30k
.venv\Scripts\python.exe -m lili_shap.run --stage pilot --output results/pilot_001
```

The downloader samples 30 unique rows from `nlphuji/flickr30k` with seed 42,
downloads only those images, and chooses one ground-truth caption per image.
The source's `test` container includes Flickr30k's original partitions; sampling
uses the entire container, not an asserted held-out retrieval split. Source rows,
population size, captions, IDs, and image hashes are saved. Ground-truth matching
means annotation pairing, not filtering for CLIP retrieval success.

Alternatively pass `--manifest your_pairs.csv` with columns
`image_id,image_path,caption`, one row per image, at least 30 rows. Relative paths
resolve from the manifest. Its first 30 rows define the experiment; its first three
define the pilot. Sample your local manifest with seed 42 before supplying it.

Inspect all three pilot `image_*/perturbations.png` and `explanation.png` figures.
Verify sensible masks, visible expansion, successful removal rather than object
reconstruction, and valid image outputs. Only after review, set
`visual_checks_passed` to `true` in the pilot's `review.json`, then run:

```powershell
.venv\Scripts\python.exe -m lili_shap.run --stage full --reviewed-pilot results/pilot_001 --output results/full_001
```

CLIP and pretrained TorchScript big-LaMa weights download on first use.
`--lama-weights` selects an existing checkpoint. `--device cpu` is supported but
slow. Images retain aspect ratio and are limited to a 512-pixel longest side
before segmentation; the 3-pixel dilation refers to this saved working image.
CLIP applies its standard resize/crop preprocessing. All run directories refuse
overwrites. Both stages emit `results.csv`, `summary.json`, raw arrays, curves,
configuration, package versions, model provenance, timing, and inspection images.
Summary p-values from a 3-image pilot are diagnostic, not evidence for H1.
TF32 is disabled so the single-image original score agrees with batched full
coalitions to float32 tolerance. Qualitative perturbations remove the region
ranked highest by Mean-SHAP, with the same target for all methods.
TorchScript graph optimization is disabled because its profiling executor can
change LaMa's output after the first call for a new image shape. Pilot checks
also require LaMa and expanded LaMa to agree on the identical all-removed mask.

## Estimator and evaluation conventions

Kernel SHAP uses the Shapley kernel weighted linear regression, with exact empty
and full endpoint constraints. The mandatory singletons and complements are
included once; remaining unique coalitions are uniformly sampled from the binary
cube. The same matrix is reused across all methods. This is a 128-coalition
approximation, not exact Shapley enumeration. Fit error is saved for inspection.

Deletion ranks only strictly positive SHAP values, with deterministic tie breaks.
Every cumulative Telea mask is applied to the original image, never recursively
to an already inpainted image. The x-axis is the actual fraction of pixels removed,
not feature count. A constant tail extends a positive-only curve to x=1 so AUCs
share a common domain. Raw curves and positive-area fractions are saved; this
convention matters when methods assign different numbers of positive features.
Scores are raw cosine similarities, without confidence conversion or normalization.

Telea has no context for a completely masked image. That endpoint explicitly uses
the original global RGB mean for all methods. LaMa is called on the fully masked
empty coalition, as specified. Unmasked pixels are preserved exactly and outputs
are cropped back after padding to multiples of eight.

Paired differences favor the second method when positive. Wilcoxon tests are
two-sided, uncorrected exploratory tests; exact ties across all images return
p=1. Automated checks do not certify visual realism or scientific faithfulness.
No automatic promising/not-promising verdict is assigned without qualitative review.

Implementation references: [Hugging Face CLIP](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/clip),
[LaMa](https://github.com/advimman/lama),
and the [TorchScript checkpoint wrapper](https://github.com/enesmsahin/simple-lama-inpainting).

## Delivered experiment

The validated 30-image run is `results/full_002`, after visual review of
`results/pilot_004`. Its CSV and summary are also exported to the requested
`results/results.csv` and `results/summary.json`. `results/assessment.json`
records the exploratory interpretation; `results/run_index.json` identifies the
validated artifacts. All configuration, versions, hashes, raw arrays, curves,
timing, and figures remain in the full run directory.

LaMa's mean AUC improvement is approximately 1%, with 16/30 per-image wins and
two-sided Wilcoxon p=0.655. Expanded LaMa slightly worsens mean and median AUC
(p=0.477 against the baseline). This run provides no convincing support for H1
or H2. No additional datasets or methods were added.

Earlier pilots and `results/full_001` are preserved for diagnosis. The latter
predates the TorchScript stability fix and is excluded from conclusions; see its
`validation_note.json`. Use fresh output directories for future runs.
