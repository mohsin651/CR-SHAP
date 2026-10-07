# CR-SHAP

Contrastive Retrieval SHAP for explaining frozen CLIP image–text preference.

Pointwise SHAP explains `s(I, T+)`. CR-SHAP explains `s(I, T+) − s(I, T−)`: which image regions explain a preference over a competing caption. Both targets share perturbations and inference. No model training is involved.

## Results

Matched-area preference deletion AUC (lower is better):

| Dataset | CLIP backbone | n | Pointwise | CR-SHAP | Relative improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| SugarCrepe | ViT-B/32 | 699 | 0.6703 | 0.6255 | 6.69% |
| Flickr30k, complete eligible population | ViT-B/32 | 793 | 0.6914 | 0.6238 | 9.78% |
| SugarCrepe | ViT-B/16 | 200 | 0.6822 | 0.6310 | 7.50% |
| Flickr30k | ViT-B/16 | 200 | 0.7242 | 0.6625 | 8.52% |

See the [master results record](CR_SHAP_MASTER_RESULTS.md) for confidence intervals, source keys, protocols, and limitations. Exact values remain in the linked JSON summaries.

The datasets are analyzed separately. Sampling and model-selected competitors differ across studies. The 793-image Flickr30k analysis extends the original 300-image study. Results concern CLIP preference under Telea deletion, not proof of causal faithfulness or universal superiority. Grounding evaluation found lower Pointing Game accuracy for CR-SHAP (78.60% versus 83.61%) and an uncertain EIB difference.

## Repository contents

| Directory | Purpose |
| --- | --- |
| `lili_shap/` | Shared CLIP, SLIC, constrained Kernel SHAP, masking, and original LaMa study |
| `experiment2/` | SugarCrepe discovery and contrastive targets |
| `experiment3/` | SugarCrepe confirmation and matched-area evaluation |
| `experiment4/` | Natural Flickr30k retrieval evaluation |
| `experiment4r/` | Complete eligible Flickr30k population extension |
| `experiment5/` | Pointwise and contrastive Grad-ECLIP comparison |
| `experiment6/` | Flickr30k Entities grounding using saved explanations |
| `experiment7/` | ViT-B/16 replication on both datasets |
| `tests/` | Estimator, evaluator, geometry, and reproducibility tests |
| `results/` | Curated reports, summaries, metric tables, and provenance |

Datasets, checkpoints, virtual environments, downloaded third-party reference files, logs, and bulky raw per-example artifacts are excluded. Published metadata preserves references to original local artifacts; those raw files are not all distributed here. Historical metadata may contain machine-specific paths.

## Setup

Use Python 3.11:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS use `.venv/bin/python`. GPU execution requires a compatible PyTorch/CUDA build. `requirements-lock.txt` records the original environment, including a CUDA-specific PyTorch build.

## Reproduction

Start with the [original setup and pilot instructions](LILI_README.md). Each experiment module contains its preparation, screening, execution, reporting, or verification steps; the corresponding saved report documents the frozen protocol and exclusions. Download datasets separately under their access terms.

Full downstream reproduction requires raw upstream outputs. Aggregate tables cannot replace images, coalition arrays, or attribution maps. Several pipelines use fixed paths and reject overwrites. For a fresh run, use a separate checkout and move that checkout's published `results/` directory aside first. Retain this checkout's published evidence for comparison. Review pilots as required by each protocol before full execution.

The shared estimator uses approximately 16 SLIC regions, 128 unique coalitions, mean RGB masking, and constrained Kernel SHAP. Deletion applies cumulative whole-region masks to the original image with Telea radius 3. Matched-area evaluation ranks all signed values and integrates observed 10–50% crossings. Positive-only evaluation uses a constant integration tail. These endpoints differ from the original LaMa raw-cosine AUC.

## References

- [CLIP](https://github.com/openai/CLIP)
- [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe)
- [Flickr30k Entities](https://github.com/BryanPlummer/flickr30k_entities)
- [Grad-ECLIP paper](https://proceedings.mlr.press/v235/zhao24p.html) and [official implementation](https://github.com/Cyang-Zhao/Grad-Eclip)
- [LaMa](https://github.com/advimman/lama)

The Grad-ECLIP comparison uses the model-preserving **multi-head variant** (Appendix C), preserving the frozen CLIP function. It is not a numerical reproduction of the default single-head demo. Third-party models and datasets retain their respective licenses and terms.
