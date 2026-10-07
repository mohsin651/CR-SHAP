"""Render a self-contained report after inspecting the fixed experiment results."""
import argparse
import json
from pathlib import Path


def report(root, classification, rationale):
    root = Path(root)
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    rows = json.loads((root/'rows.json').read_text(encoding='utf-8'))
    screening = summary['screening']
    overall = summary['overall']
    checks = json.loads((root/'checks.json').read_text(encoding='utf-8'))
    allocation = screening['allocation']
    lines = ['# Experiment 2: Contrastive Retrieval SHAP for CLIP', '',
        '## Question and controlled setup', '',
        'Does explaining the positive-minus-hard-negative cosine margin identify image regions whose removal destroys CLIP’s positive-caption preference faster than explaining positive-caption cosine alone?', '',
        'Frozen `openai/clip-vit-base-patch32`, Hugging Face Transformers. Pointwise SHAP explains `cosine(I,T+)`; CR-SHAP explains `cosine(I,T+) - cosine(I,T−)`. Both use the unchanged Experiment 1 constrained Kernel SHAP solver, 128 identical seeded coalitions, the same mean-masked images, and one image embedding per coalition for both caption scores. No LaMa, mask expansion, training, or additional method was used.', '',
        'SLIC settings: `n_segments=16`, `compactness=30`, `start_label=0`, `channel_axis=-1`; all other parameters retain scikit-image 0.24.0 defaults. Actual segment counts were saved. Images use EXIF transpose, RGB conversion, aspect-preserving maximum side 512, then the same validated CLIP processor. Seed: 42. Model commit: `'+screening['clip_commit']+'`.', '',
        '## Screening and sampling', '',
        f"All **{screening['overall']['n_examined']:,}** official [SugarCrepe](https://github.com/RAIVNLab/sugar-crepe) instances were screened: **{screening['overall']['n_correct']:,} correctly ranked**, **{screening['overall']['n_incorrect']:,} incorrectly ranked**, **{screening['overall']['n_ties']} ties**, accuracy **{100*screening['overall']['accuracy']:.2f}%**. A seeded, balanced sample of 100 correctly ranked instances was selected before generating explanations. Official category names were preserved. No instance was replaced based on explanation performance.", '',
        '| Category | Examined | Correct | Incorrect | Accuracy | Selected |',
        '|---|---:|---:|---:|---:|---:|']
    for c,a in screening['categories'].items():
        lines.append(f"| {c} | {a['n_examined']} | {a['n_correct']} | {a['n_incorrect']} | {100*a['accuracy']:.2f}% | {allocation[c]} |")
    lines += ['', '## Evaluation convention', '',
        f"Both rankings use the same validated Telea inpainting evaluator (`cv2.INPAINT_TELEA`, radius 3), applying cumulative masks to the original image. Only strictly positive SHAP regions are deleted, in descending order, with ascending region-index ties. Pairwise preference is `sigmoid(gamma × (cosine_positive − cosine_negative))`, where pretrained **gamma = {summary['gamma']:.6f}**. PDAUC integrates preference against actual removed pixel fraction; lower is better.", '',
        'Per the user’s clarification, after all positive regions are deleted the final preference is held constant to x=1 **solely for AUC integration**. Zero and negative regions are never deleted. With no positive regions, original preference is constant over [0,1]. Unreached 25%/50% thresholds are missing, not extrapolated; rates show coverage and a comparison restricted to instances reached by both methods. The validated global-mean fallback is retained only when Telea has no unmasked context.', '',
        '## Overall results', '',
        '| Metric | Pointwise SHAP | CR-SHAP |', '|---|---:|---:|',
        f"| Mean PDAUC | {overall['pointwise_mean_pdauc']:.6f} | {overall['cr_shap_mean_pdauc']:.6f} |",
        f"| Median PDAUC | {overall['pointwise_median_pdauc']:.6f} | {overall['cr_shap_median_pdauc']:.6f} |",
        f"| Per-instance wins | {overall['pointwise_wins']} | {overall['cr_shap_wins']} |",
        f"| No positive attribution | {overall['no_positive_attribution']['pointwise']} | {overall['no_positive_attribution']['cr_shap']} |", '',
        f"n={overall['n']}; ties={overall['ties']}. Mean paired PDAUC difference (Pointwise − CR): **{overall['mean_paired_difference']:+.6f}**; median paired difference: **{overall['median_paired_difference']:+.6f}**. Relative mean improvement: **{overall['relative_mean_improvement_percent']:+.2f}%**. Positive differences favor CR-SHAP.", '',
        f"Paired two-sided Wilcoxon: statistic **{overall['wilcoxon']['statistic']:.3f}**, p=**{overall['wilcoxon']['pvalue']:.6g}**. Paired percentile bootstrap 95% CI for the mean difference: **[{overall['paired_bootstrap_mean_difference_ci95']['low']:+.6f}, {overall['paired_bootstrap_mean_difference_ci95']['high']:+.6f}]**, using 10,000 resamples and seed 42.", '',
        '| Threshold | Pointwise flip rate (reached n) | CR flip rate (reached n) | Common n | Pointwise / CR on common cases |',
        '|---|---:|---:|---:|---:|']
    def rate(value):
        return 'missing' if value is None else f'{value:.2f}%'
    for t in ['25','50']:
        a=overall['rankflip'][t]
        common=a['paired_common_coverage']
        lines.append(f"| RankFlip@{t} | {rate(a['pointwise']['percent_among_reached'])} ({a['pointwise']['n_reached']}) | {rate(a['cr_shap']['percent_among_reached'])} ({a['cr_shap']['n_reached']}) | {common['n']} | {rate(common['pointwise'])} / {rate(common['cr_shap'])} |")
    lines += ['', '| Metric at actual ≥25% deletion | Pointwise | CR-SHAP |', '|---|---:|---:|']
    for key,label in [('mean_margin','Mean margin (lower better)'),('mean_margin_drop','Mean margin drop (higher better)')]:
        a=overall['margin25']
        values=[f"{a[m][key]:.6f}" if a[m][key] is not None else 'missing' for m in ['pointwise','cr_shap']]
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    lines += ['', '## Category results', '',
        '| Category | n | Point mean | CR mean | Mean paired difference | Relative improvement | CR / Point wins / ties |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c,a in summary['categories'].items():
        lines.append(f"| {c} | {a['n']} | {a['pointwise_mean_pdauc']:.6f} | {a['cr_shap_mean_pdauc']:.6f} | {a['mean_paired_difference']:+.6f} | {a['relative_mean_improvement_percent']:+.2f}% | {a['cr_shap_wins']} / {a['pointwise_wins']} / {a['ties']} |")
    lines += ['', '| Category | Point / CR RankFlip@25 (coverage) | Point / CR RankFlip@50 (coverage) |', '|---|---:|---:|']
    for c,a in summary['categories'].items():
        cells=[]
        for t in ['25','50']:
            rates=a['rankflip'][t]
            cells.append(' / '.join(f"{rate(rates[m]['percent_among_reached'])} ({rates[m]['n_reached']}/{a['n']})" for m in ['pointwise','cr_shap']))
        lines.append(f'| {c} | {cells[0]} | {cells[1]} |')
    correlation=summary['original_margin_relationship']
    lines += ['', 'No category-level significance tests were performed on these small strata. Common-case category flip rates are also available in `summary.json`.', '',
        '## Exploratory margin relationship and visualizations', '',
        f"Spearman correlation between original cosine margin and Pointwise − CR PDAUC difference: rho={correlation['spearman_rho']:.6f}, p={correlation['pvalue']:.6g}. This analysis is exploratory; see `margin_relationship.png`.", '',
        'Fifteen figures were selected strictly by PDAUC difference: five largest improvements, five closest-to-zero remaining differences, and five largest worsenings. IDs and the deterministic selection rule are in `figure_selection.json`. These are outcome-stratified diagnostic examples, not a representative estimate of average visual quality.', '',
        '## Validation, timing, and interpretation', '',
        f"The five-instance pilot passed before the full run. Five Experiment 2 unit tests passed. All {checks['n']} final examples passed sanity checks, including full-coalition scores, correct caption embeddings, shared masks/images, frozen model state, efficiency, strict positive deletion ordering, preference calculation, and Shapley linearity. Maximum absolute linearity error: **{checks['max_abs_linearity_error']:.3g}**. Experiment 1 file hashes were verified unchanged.", '',
        f"Mean recorded explanation runtime: Pointwise **{np_mean(rows,'pointwise_runtime_seconds'):.3f} s**, CR **{np_mean(rows,'cr_shap_runtime_seconds'):.3f} s**. These are shared coalition generation/scoring time plus each target’s own regression time; shared inference actually ran once, not twice. Raw timing fields preserve this distinction.", '',
        f'**Research classification: {classification}.** {rationale}', '',
        f"Limitations: one model, one seed, 100 correctly ranked instances drawn from {summary['selected_unique_images']} distinct images; the sample is balanced by category, not weighted to benchmark frequency. Repeated images mean instance-level Wilcoxon and bootstrap calculations do not establish independence at image level. Positive-only tail integration and differing positive-area coverage can influence PDAUC. Telea preference deletion is a controlled proxy, not proof of causal faithfulness. Category effects require replication; no parameters were optimized after seeing results.", '',
        'All raw evidence is retained in `example_*/raw.npz`, `deletion_curves.json`, and associated metadata. Screening evidence and cached normalized embeddings are in `data/sugarcrepe_screening`. Experiment 1 was neither modified nor rerun.']
    with (root/'report.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines)+'\n')


def np_mean(rows,key):
    return sum(r[key] for r in rows)/len(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/experiment_2')
    parser.add_argument('--classification',required=True,choices=['VERY PROMISING','PROMISING','MIXED','NOT PROMISING'])
    parser.add_argument('--rationale',required=True)
    args=parser.parse_args()
    report(args.output,args.classification,args.rationale)
