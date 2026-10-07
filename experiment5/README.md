# Experiment 5: objective-controlled SHAP and Grad-ECLIP comparison

Uses the exact Experiment 4 sample, captions, SLIC, model weights, preprocessing,
and evaluator. Experiment 4 SHAP metrics are copied verbatim, never recomputed.

The official Grad-ECLIP notebook changes the last attention layer to one head.
This experiment requires the original twelve-head CLIP function. We therefore
use the paper's multi-head variant (Appendix C), retaining the original forward
pass and the notebook's concatenated-channel query/key spatial weights and
gradient/value relevance formula. It is explicitly not a numerical reproduction
of the single-head demo. Target: last visual block, index 11; no layer tuning.

The official ReLU is retained for both targets. Contrastive gradients come from
the raw positive-minus-negative cosine scalar, before ReLU, rather than from a
subtraction of two positive heatmaps. Pixel maps undo CLIP's actual center crop;
unseen working-image areas receive zero relevance. Region scores are pixel means.

```powershell
.venv\Scripts\python.exe -m pytest tests/test_experiment5.py -q
.venv\Scripts\python.exe -m experiment5.run --stage pilot
.venv\Scripts\python.exe -m experiment5.verify results/experiment_5_pilot
.venv\Scripts\python.exe -m experiment5.run --stage full
.venv\Scripts\python.exe -m experiment5.verify
.venv\Scripts\python.exe -m experiment5.report
```

All outputs refuse overwrites. Full runs require a passed matching five-image
pilot. Parameter hashes and prior-artifact hashes are checked before and after.
No training, optimizer, sample replacement, method tuning, or Experiment 6.

References: [ICML paper](https://proceedings.mlr.press/v235/zhao24p.html),
[pinned official notebook](https://github.com/Cyang-Zhao/Grad-Eclip/blob/e370e6cb194faf2020f5d1ed268f9d57e91a38e6/grad_eclip_image.ipynb).
