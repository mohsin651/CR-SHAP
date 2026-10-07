# Experiment 3: frozen replication

This isolated package reuses Experiment 2 model encoding, image loading, SHAP
estimation, positive-only deletion, and Experiment 1 coalition/Telea components.
It leaves both earlier experiments unchanged. No LaMa model is instantiated.

From the workspace with the existing environment and official SugarCrepe files:

```powershell
.\.venv\Scripts\python.exe -m experiment2.screen --output data/sugarcrepe_screening_exp3
.\.venv\Scripts\python.exe -m experiment3.prepare
.\.venv\Scripts\python.exe -m experiment3.preflight
.\.venv\Scripts\python.exe -m pytest tests/test_experiment3.py -q
.\.venv\Scripts\python.exe -m experiment3.run --stage pilot
.\.venv\Scripts\python.exe -m experiment3.verify results/experiment_3_pilot
.\.venv\Scripts\python.exe -m experiment3.run --stage full
.\.venv\Scripts\python.exe -m experiment3.verify
.\.venv\Scripts\python.exe -m experiment3.report
```

Outputs refuse overwrites. `prepare` checks the exact expected screening counts,
model commit, and gamma before freezing selection. It excludes all discovery
instance IDs, uses seed 2026, and selects up to 100 unique images per category,
uniformly choosing one eligible instance per selected image. Only if there are
too few images does it fill with unused instances; shortages are documented.
Image repetition across categories is allowed. SHAP seed remains 42.
The segmentation preflight checks the frozen sample supports 128 unique
coalitions. A failure stops the runner and requires an explicitly authorized
protocol amendment; it never changes parameters or replaces examples.

The model and processor are explicitly pinned to the validated commit. The
seven-instance pilot validates the same protocol, with one instance per category.
Model mismatches, failures, and invalid coalitions stop execution without sample
replacement. The full run requires a matching passed pilot and selection hash.

Positive-only PDAUC exactly preserves the Experiment 2 convention. Matched-area
evaluation sorts every signed attribution descending, ties by region index. It
uses whole cumulative SLIC masks and the first state reaching each budget. It
never uses absolute values or clips signed rankings. Telea uses the original
image for each cumulative mask, radius 3, with the unchanged global-mean fallback
when all pixels are removed. The normalized matched AUC includes the original
and five budget crossing states, integrates over actual area, and has no tail.
Duplicate budget states are retained with zero-width intervals.

All positive differences favor CR: Pointwise minus CR for AUC, preference and
margin; CR minus Pointwise for margin drop and rank-flip rate. Results include
both method values to make signs inspectable. Instance and image-cluster
percentile bootstrap intervals use 10,000 draws, seed 42. Cluster draws include
all instances per drawn image and preserve cluster multiplicity. Wilcoxon tests
are paired, two-sided; exact McNemar tests use a binomial test on discordant
instance outcomes. These instance tests do not themselves model image dependence.
The primary endpoint is positive-only PDAUC. Matched AUC is the main robustness
endpoint; other budgets/categories are secondary with unadjusted p-values.

Per-example raw evidence saves segment maps, binary coalition matrices (pixel
masks reconstruct exactly from segmentation), hashes, both cosine caches, SHAP
vectors, positive curves, full signed rankings, and every matched cumulative
pixel mask. `verify` independently reconstructs coalition hashes and validates
rankings, masks, first crossings and formulas without rerunning CLIP.
Integrity snapshots protect prior source/result files. Shared inference runtime
and the two regression runtimes are recorded separately.

The provided request ends abruptly at “exact McNem” in section 30. The implemented
test interprets this as exact McNemar; any subsequent instructions require the
remaining attachment text. No methodological choices are tuned using outcomes.

## Authorized amendment and continuation

The original 700-instance segmentation preflight found one six-region case,
`swap_att:39`, incompatible with 128 unique coalitions. After the user requested
fixing and running, `python -m experiment3.amend` archived the original full and
pilot attempts and excluded that one instance without replacement. The final
sample is 699 (99 swap_att, 100 in each other category). Method settings remain
unchanged. The original selection/preflight files have `.original_700` copies;
the timing and reason for the amendment are recorded in `protocol_amendment.json`.

Continuation uses a fresh matching pilot, then:

```powershell
.\.venv\Scripts\python.exe -m experiment3.run --stage full --reuse-completed results/experiment_3_stopped_001
```

This copies the 55 previously completed and independently validated examples
after checking the method configuration, selected-instance prefix and checks.
It computes all remaining instances and validates the complete final evidence.
