"""Render results without modifying any experimental decisions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment2.io import save_json, sha256, write_csv
from .core import BUDGETS


def main():
    root = Path('results/experiment_3')
    summary = json.loads((root/'summary.json').read_text())
    selection = json.loads((root/'selection_report.json').read_text())
    checks = json.loads((root/'checks.json').read_text())
    verification = json.loads((root/'verification.json').read_text())
    rows = json.loads((root/'rows.json').read_text())
    primary,robust = summary['primary'],summary['robustness']
    def interval(s,key='image_cluster_bootstrap_ci95'):
        ci = s[key]
        return f"[{ci['low']:+.6f}, {ci['high']:+.6f}]"
    def supported(s):
        return s['mean_difference']>0 and s['wilcoxon']['pvalue']<.05 and s['image_cluster_bootstrap_ci95']['low']>0
    conclusion = ('The positive-only advantage replicated, and the matched-area endpoint supports the same direction.' if supported(primary) and supported(robust)
        else 'The positive-only advantage replicated, but the matched-area endpoint does not provide the same statistical support.' if supported(primary)
        else 'The pre-declared positive-only advantage did not receive clear confirmatory support in this sample.')
    lines = ['# Experiment 3: confirmatory CR-SHAP evaluation on SugarCrepe','',conclusion,'',
        'This run tests the previously observed advantage under the specified frozen pipeline. Statistical support below means a positive mean paired difference, a two-sided Wilcoxon p < 0.05, and an image-cluster 95% interval above zero; these are reported explicitly rather than as proof of causal faithfulness.','',
        '## Frozen design and sample','',
        'Official SugarCrepe screening reproduced **7,511 instances, 5,745 correctly ranked, 1,766 incorrectly ranked, and zero ties** (76.49% pair accuracy). The new screening recomputed embeddings and scores rather than using the old counts.','',
        f"Sampling seed 2026 initially selected **700 correctly ranked instances**, 100 in every official category. All **{selection['excluded_experiment2_instances']} Experiment 2 example IDs** were excluded, with no eligible-pool shortages or reintroductions. The final sample contains **{selection['n']} instances: 99 swap_att and 100 in each other category** after the authorized technical exclusion described below. Sampling used at most one instance per image within each category: uniformly sampled image IDs, then a uniformly sampled eligible instance for each image. Thus sampling is uniform over images within category, not uniform over caption instances. Cross-category image reuse was allowed as specified.",'',
        'Protocol amendment: the selected instance `swap_att:39` produced six regions under frozen SLIC, which permit only 64 unique binary coalitions. The unchanged sampler requires 128 unique coalitions. After an outcome-independent check of all 700 segmentations, the original run was stopped at 55 completed instances. The user authorized fixing and running after the proposed single technical exclusion. This instance was excluded without replacement; no method settings changed. The exclusion was made after some outcomes had been computed, but its sole criterion was coalition feasibility. Original selections and partial results remain archived in `results/experiment_3_stopped_001`, and the original pilot in `results/experiment_3_pilot_original`. The 55 independently verified completed instances were reused verbatim; the remaining instances were evaluated under the same frozen configuration.','',
        f"There are **{selection['unique_images']} unique images**, **{selection['images_repeated']} images used more than once**, and at most **{selection['maximum_instances_per_image']} instances for one image**. Exclusion applies to instance IDs, not all images seen in Experiment 2; {len({r['image_id'] for r in rows}&{r['image_id'] for r in json.loads(Path('results/experiment_2/selected_instances.json').read_text())})} images also occurred there. The image-cluster bootstrap accounts for image repetition within Experiment 3.",'',
        'Frozen `openai/clip-vit-base-patch32` at commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, eval mode, gamma 100.0. Images use EXIF transpose, RGB, aspect-preserving maximum side 512, then the validated CLIP processor. SLIC uses 16 requested segments, compactness 30, start label 0, channel axis −1, and scikit-image 0.24.0 defaults. The unchanged constrained Kernel SHAP estimator uses 128 unique coalitions and seed 42 with mean RGB masking. One image embedding supplies both caption scores per coalition. The same positive and official negative captions are used throughout. No model training or parameter tuning occurred.','',
        '## Endpoints and paired inference','',
        '| Endpoint (lower is better) | Pointwise mean / median | CR mean / median | Mean difference | CR wins | Wilcoxon p | Image-cluster 95% CI |',
        '|---|---:|---:|---:|---:|---:|---|']
    for label,s in [('Primary: positive-only PDAUC',primary),('Robustness: MatchedArea-PDAUC-50',robust)]:
        lines.append(f"| {label} | {s['point_mean']:.6f} / {s['point_median']:.6f} | {s['cr_mean']:.6f} / {s['cr_median']:.6f} | {s['mean_difference']:+.6f} | {s['cr_wins']}/{s['n']} | {s['wilcoxon']['pvalue']:.6g} | {interval(s)} |")
    lines += ['',f"Primary relative mean improvement: **{primary['relative_mean_improvement_percent']:.2f}%**; median paired difference {primary['median_difference']:+.6f}; Pointwise wins {primary['point_wins']}, ties {primary['ties']}. Wilcoxon statistic {primary['wilcoxon']['statistic']:.6g}. Paired instance-bootstrap CI {interval(primary,'paired_bootstrap_ci95')}.",'',
        f"Matched-area relative mean improvement: **{robust['relative_mean_improvement_percent']:.2f}%**; median paired difference {robust['median_difference']:+.6f}; Pointwise wins {robust['point_wins']}, ties {robust['ties']}. Wilcoxon statistic {robust['wilcoxon']['statistic']:.6g}. Paired instance-bootstrap CI {interval(robust,'paired_bootstrap_ci95')}.",'',
        'Both bootstraps use 10,000 resamples, seed 42, and percentile intervals. The cluster bootstrap draws the original number of image IDs with replacement, includes every instance belonging to each drawn image (including repeats of that cluster), and computes an instance-weighted mean difference. Cluster intervals are the preferred uncertainty estimates. Wilcoxon is the requested paired instance test and does not itself model image clustering.','',
        'Positive-only PDAUC reproduces Experiment 2: delete strictly positive regions in descending SHAP order, index ties ascending, using Telea radius 3 on the original image with each cumulative mask. Extend the final preference horizontally to x=1 for integration only; no positives means a constant original preference. Zero and negative regions are never removed in this evaluation.','',
        'Matched-area deletion ranks **all signed SHAP values**, descending with index ties ascending. For each 10–50% target it uses the first whole-region state reaching or exceeding the target. MatchedArea-PDAUC-50 integrates the original preference and these five observed states against their actual pixel fractions, dividing by the final fraction. Repeated crossing states contribute zero-width intervals. There is no extrapolation to 100%. Telea and the validated fully masked global-mean fallback remain unchanged.','',
        '## Fixed removal budgets','',
        '| Budget | Actual area Point / CR | Preference Point / CR | Preference difference | Margin Point / CR | Margin drop Point / CR | Drop difference | Rank flip Point / CR | Exact McNemar p |',
        '|---|---|---|---:|---|---|---:|---|---:|']
    budget_rows = []
    for q in BUDGETS:
        entry = summary['budgets'][str(q)]
        pref,drop = entry['preference'],entry['margin_drop']
        pm = np.mean([r[f'point_margin{q}'] for r in rows])
        cm = np.mean([r[f'cr_margin{q}'] for r in rows])
        lines.append(f"| {q}% | {entry['point_mean_actual_area']:.3%} / {entry['cr_mean_actual_area']:.3%} | {pref['point_mean']:.6f} / {pref['cr_mean']:.6f} | {pref['mean_difference']:+.6f} | {pm:+.6f} / {cm:+.6f} | {drop['point_mean']:.6f} / {drop['cr_mean']:.6f} | {drop['mean_difference']:+.6f} | {entry['point_rankflip_rate']:.2%} / {entry['cr_rankflip_rate']:.2%} | {entry['exact_mcnemar_pvalue']:.6g} |")
        budget_rows.append({'target_budget':q/100, **{k:v for k,v in entry.items() if not isinstance(v,dict)},
            'preference_difference':pref['mean_difference'],'margin_drop_difference':drop['mean_difference'],
            **entry['rankflip_2x2']})
    lines += ['', 'Every instance reaches every matched-area budget. Whole-region overshoot means actual areas can differ between methods; this is the specified approximate budget matching, not exact equal-pixel removal.','',
        '| Budget | Neither flipped | Pointwise only | CR only | Both flipped | Preference cluster CI | Margin-drop cluster CI |',
        '|---|---:|---:|---:|---:|---|---|']
    for q in BUDGETS:
        entry = summary['budgets'][str(q)]
        counts = entry['rankflip_2x2']
        lines.append(f"| {q}% | {counts['both_not_flipped']} | {counts['point_only_flipped']} | {counts['cr_only_flipped']} | {counts['both_flipped']} | {interval(entry['preference'])} | {interval(entry['margin_drop'])} |")
    lines += ['', 'The exact two-sided McNemar test is the binomial test with probability 0.5 on the discordant paired outcomes. It operates on instance pairs and does not adjust for shared-image dependence. Fixed-budget and category tests are secondary and unadjusted for multiple comparisons; see `summary.json` for all paired bootstrap intervals, Wilcoxon statistics, and p-values.','',
        '## Category results','',
        '| Category | n | Positive PDAUC Point / CR | Positive difference | Matched PDAUC Point / CR | Matched difference |',
        '|---|---:|---|---:|---|---:|']
    category_rows = []
    for c,entry in summary['categories'].items():
        a,b = entry['primary'],entry['robustness']
        lines.append(f"| {c} | {a['n']} | {a['point_mean']:.6f} / {a['cr_mean']:.6f} | {a['mean_difference']:+.6f} | {b['point_mean']:.6f} / {b['cr_mean']:.6f} | {b['mean_difference']:+.6f} |")
        category_rows.append({'category':c,'n':a['n'],'positive_difference':a['mean_difference'],'matched_difference':b['mean_difference'],
            'positive_cluster_ci_low':a['image_cluster_bootstrap_ci95']['low'],'positive_cluster_ci_high':a['image_cluster_bootstrap_ci95']['high'],
            'matched_cluster_ci_low':b['image_cluster_bootstrap_ci95']['low'],'matched_cluster_ci_high':b['image_cluster_bootstrap_ci95']['high']})
    lines += ['', '## Validation, interpretation, and evidence','',
        f"Three new unit tests passed. A seven-instance pilot covered every category and passed before the full run. All {checks['n']} final instances passed checks; independent reconstruction validated {verification['shared_coalitions_validated']:,} coalition masks/images by hashes plus saved deletion masks, rankings, first crossings, and metric formulas. Maximum Shapley linearity error: {verification['max_abs_linearity_error']:.3g}. All {verification['protected_files_unchanged']:,} protected Experiment 1/2 source and result files remained unchanged.",'',
        f"Positive-only summaries: mean positive area Pointwise {np.mean([r['point_positive_area_fraction'] for r in rows]):.3%}, CR {np.mean([r['cr_positive_area_fraction'] for r in rows]):.3%}; mean positive-region count Pointwise {np.mean([r['point_num_positive_regions'] for r in rows]):.2f}, CR {np.mean([r['cr_num_positive_regions'] for r in rows]):.2f}; no-positive cases Pointwise {sum(r['point_no_positive_attribution'] for r in rows)}, CR {sum(r['cr_no_positive_attribution'] for r in rows)}.",'',
        'The comparison concerns CLIP preference under Telea perturbations. It does not establish causal faithfulness beyond that evaluator. Conditioning on correctly ranked examples, equal category weighting, approximate Kernel SHAP, whole-region budget overshoot, one model and sampling seed, and residual image dependence limit generalization. The matched-area evaluation addresses the positive-only constant-tail concern but still uses different removal states and may end slightly beyond 50%. Excluding discovery instance IDs does not guarantee disjoint images. No outcome-based sample replacement or method tuning was performed.','',
        f"Raw evidence is in `example_000` through `example_{len(rows)-1:03d}`: original images, segment maps, shared binary coalitions and hashes, positive/negative coalition scores, all three SHAP vectors, regression diagnostics, positive-only curves, full signed orders, actual cumulative matched masks, cosine scores, margins, preferences, and validation checks. The complete per-instance table is `results.csv`; inference time is recorded once as shared cost, with Pointwise and CR regression times separate. Source revision, selection, package versions, screening counts, hashes, and integrity evidence are saved alongside the metrics.",'',
        'The supplied request ended at “exact McNem” in section 30; this was interpreted as exact McNemar. No additional requirements beyond the supplied text were assumed.','']
    report = root/'report.md'
    if report.exists(): raise FileExistsError(report)
    report.write_text('\n'.join(lines),encoding='utf-8')
    write_csv(root/'budget_results.csv',budget_rows)
    write_csv(root/'category_results.csv',category_rows)
    fig,axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for prefix,label in [('point','Pointwise'),('cr','CR-SHAP')]:
        areas = [0.] + [np.mean([r[f'{prefix}_actual_area{q}'] for r in rows]) for q in BUDGETS]
        values = [np.mean([r['original_preference'] for r in rows])] + [np.mean([r[f'{prefix}_preference{q}'] for r in rows]) for q in BUDGETS]
        axes[0].plot(areas,values,'o-',label=label)
        axes[1].plot(BUDGETS,[100*np.mean([r[f'{prefix}_rankflip{q}'] for r in rows]) for q in BUDGETS],'o-',label=label)
    axes[0].set(xlabel='Mean actual fraction removed',ylabel='Mean preference',title='Observed matched-area states')
    axes[1].set(xlabel='Nominal removal budget (%)',ylabel='Rank flip (%)',title='Paired retrieval failures')
    for ax in axes: ax.legend(); ax.grid(alpha=.25)
    fig.savefig(root/'matched_budget_summary.png',dpi=180)
    plt.close(fig)
    save_json(root/'implementation_hashes.json',{str(p):sha256(p) for p in sorted(Path('experiment3').glob('*')) if p.is_file()})
    print(conclusion)


if __name__=='__main__':
    main()
