"""Complete fixed Experiment 5 report; no follow-up experiment launch."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries

from experiment2.io import save_json, write_csv
from experiment3.core import BUDGETS
from .core import COMPARISONS, METHODS, paired
from .run import CONFIG, EXP4

LABELS = {'shap_point':'Pointwise SHAP','shap_cr':'CR-SHAP','grad_point':'Grad-ECLIP (multi-head)',
          'grad_cr':'Contrastive Grad-ECLIP (multi-head)'}


def interval(s):
    ci = s['paired_bootstrap_ci95']; return f"[{ci['low']:+.6f}, {ci['high']:+.6f}]"


def supported(s):
    return s['mean_difference']>0 and s['wilcoxon']['pvalue']<.05 and s['paired_bootstrap_ci95']['low']>0


def main():
    root = Path('results/experiment_5')
    if (root/'report.md').exists(): raise FileExistsError(root/'report.md')
    rows = json.loads((root/'rows.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    verify = json.loads((root/'verification.json').read_text())
    if not verify['all_passed'] or verify['n']!=300: raise ValueError('Complete verified run required')
    comparisons = summary['comparisons']
    b = comparisons['B_GradECLIP_objective']['primary']; d = comparisons['D_contrastive_families']['primary']
    banswer = ('Yes: the contrastive objective improves the primary endpoint with paired statistical support.' if supported(b) else
        'The contrastive objective does not show clear paired statistical support for improvement on the primary endpoint.')
    families = ('Contrastive Grad-ECLIP has lower observed mean matched-area PDAUC than CR-SHAP.' if d['mean_difference']>0 else
        'CR-SHAP has lower observed mean matched-area PDAUC than Contrastive Grad-ECLIP.' if d['mean_difference']<0 else 'The contrastive methods have equal mean matched-area PDAUC.')
    lines = ['# Experiment 5: external Grad-ECLIP baseline and contrastive objective', '', banswer, '', families, '',
        '## Scientific question and implementation', '',
        'Does explaining relative caption preference help an established gradient explanation family, or is the observed benefit specific to SHAP? Four configurations share the exact 300-image Experiment 4 sample and evaluator. SHAP results are reused verbatim.', '',
        'This is the **model-preserving multi-head Grad-ECLIP variant**, documented in Appendix C of the [ICML 2024 paper](https://proceedings.mlr.press/v235/zhao24p.html). The [official notebook](https://github.com/Cyang-Zhao/Grad-Eclip/blob/e370e6cb194faf2020f5d1ed268f9d57e91a38e6/grad_eclip_image.ipynb) is pinned at commit `e370e6cb194faf2020f5d1ed268f9d57e91a38e6` and archived locally. Its default final-block single-head rewrite changes the model function; using that default would violate this experiment’s frozen-model requirement. Consequently these results must not be described as a numerical reproduction of the default single-head demo.', '',
        'The target is visual transformer block 11 of 12. Its twelve original attention heads remain intact. The activation is the concatenated CLS attention context before its output projection. Channel weights are its target-score gradient. Spatial weights follow the notebook: min-max-normalized cosine of the unscaled CLS query and each patch key over concatenated channels. Patch relevance is ReLU of the channel sum of gradient × patch value × spatial weight. This preserves Grad-ECLIP’s value features and spatial/channel weighting, rather than substituting Grad-CAM.', '',
        'Both methods share one forward graph. Pointwise targets raw cosine s(I,T+); contrastive targets raw cosine s(I,T+) − s(I,T−). The scalar is changed before autograd and ReLU; two positive heatmaps are not subtracted. Gradient linearity is checked before ReLU. All parameters remain frozen and eval-mode; there is no optimizer or training loop.', '',
        'The native 7×7 map is bilinearly interpolated to the processor’s 224×224 crop, placed back into its actual aspect-preserving resized canvas, and resized to the saved working image, with align_corners=False. Pixels outside the visible center crop receive zero relevance. No map normalization is applied for evaluation. SLIC relevance is the mean over all pixels in each saved region. Display heatmaps alone may be normalized.', '',
        '## Frozen sample and evaluation', '',
        'All 300 image IDs, original images, positive/negative captions, caption IDs/ownership, original cosine scores, margins, and SLIC maps are unchanged from Experiment 4. The strongest other-image caption is fixed from the complete 5,000-caption retrieval pool; no examples or negatives were replaced.', '',
        'CLIP: `openai/clip-vit-base-patch32`, commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. EXIF transpose, RGB, aspect-preserving maximum side 512, then the same processor and tokenizer. SLIC was reused, not recomputed or tuned. Seeds: 42 for deterministic runtime and bootstrap; the existing selection seed remains 2026.', '',
        'The primary endpoint is MatchedArea-PDAUC-50. All region scores are ranked descending with ascending-index ties. Whole regions are removed until the first crossing of 10/20/30/40/50% actual pixel area. Telea radius 3 is applied to the original image for each cumulative mask, with the unchanged global-mean fallback for full removal. Preference is sigmoid(gamma × raw cosine margin), gamma from the original model, approximately 100. Trapezoidal AUC uses the original state and five observed crossing states, divides by final area, and has no extrapolation. Duplicate crossings contribute zero-width intervals.', '',
        'PositiveOnly-PDAUC is secondary: strictly positive region scores are deleted, then the final preference is held constant to 100% for integration only. Grad-ECLIP’s official ReLU yields nonnegative maps, so this endpoint reflects positive spatial support, not signed SHAP contributions; its cross-family interpretation is limited.', '',
        '## Complete 2×2 headline comparison', '',
        '| Method | Objective | MatchedArea-PDAUC-50 mean / median | Relative improvement vs Pointwise SHAP | Wins / losses / ties vs Pointwise SHAP |',
        '|---|---|---|---:|---|']
    headline = []
    for method in METHODS:
        s = paired(rows,'shap_point',method,'matched_pdauc50')
        objective = 's(I,T+)' if method.endswith('point') else 's(I,T+) − s(I,T−)'
        lines.append(f"| {LABELS[method]} | {objective} | {s['second_mean']:.6f} / {s['second_median']:.6f} | {s['relative_mean_improvement_percent']:.2f}% | {s['second_wins']} / {s['first_wins']} / {s['ties']} |")
        headline.append({'method':method,'objective':objective,'mean':s['second_mean'],'median':s['second_median'],
            'improvement_vs_pointwise_shap_percent':s['relative_mean_improvement_percent'],'wins_vs_pointwise_shap':s['second_wins']})
    write_csv(root/'headline_results.csv',headline)
    lines += ['', '## Paired primary comparisons', '',
        '| Comparison (first → second) | First mean / median | Second mean / median | Mean / median difference | Relative improvement | Second / first wins / ties | Wilcoxon statistic / p | Paired mean-difference 95% CI |',
        '|---|---|---|---|---:|---|---|---|']
    primary_rows = []
    for name,c in comparisons.items():
        s = c['primary']; first,second = COMPARISONS[name]
        lines.append(f"| {name}: {LABELS[first]} → {LABELS[second]} | {s['first_mean']:.6f} / {s['first_median']:.6f} | {s['second_mean']:.6f} / {s['second_median']:.6f} | {s['mean_difference']:+.6f} / {s['median_difference']:+.6f} | {s['relative_mean_improvement_percent']:.2f}% | {s['second_wins']} / {s['first_wins']} / {s['ties']} | {s['wilcoxon']['statistic']:.6g} / {s['wilcoxon']['pvalue']:.6g} | {interval(s)} |")
        primary_rows.append({'comparison':name,**{k:v for k,v in s.items() if not isinstance(v,dict)},
            'wilcoxon_statistic':s['wilcoxon']['statistic'],'wilcoxon_p':s['wilcoxon']['pvalue'],
            'ci_low':s['paired_bootstrap_ci95']['low'],'ci_high':s['paired_bootstrap_ci95']['high']})
    write_csv(root/'primary_comparisons.csv',primary_rows)
    lines += ['', 'n=300 unique paired query images in every comparison. Positive AUC/preference differences are first minus second and favor the second method; positive margin-drop differences are second minus first. Tests are paired two-sided Wilcoxon; CIs use 10,000 paired percentile bootstrap resamples, seed 42. B is the key new objective comparison. A exactly reproduces the frozen Experiment 4 numbers. C/D and fixed budgets are secondary; p-values are unadjusted and cannot be treated as independent evidence.', '',
        '## Secondary positive-only results', '',
        '| Comparison | First mean / median | Second mean / median | Difference | Relative improvement | Second / first wins / ties | Wilcoxon p | Paired 95% CI |',
        '|---|---|---|---:|---:|---|---:|---|']
    for name,c in comparisons.items():
        s = c['positive_only']
        lines.append(f"| {name} | {s['first_mean']:.6f} / {s['first_median']:.6f} | {s['second_mean']:.6f} / {s['second_median']:.6f} | {s['mean_difference']:+.6f} | {s['relative_mean_improvement_percent']:.2f}% | {s['second_wins']} / {s['first_wins']} / {s['ties']} | {s['wilcoxon']['pvalue']:.6g} | {interval(s)} |")
    lines += ['', '## Secondary fixed budgets and RankFlip', '',
        '| Method | Budget | Mean actual area | Mean preference | Mean margin drop | RankFlip |', '|---|---:|---:|---:|---:|---:|']
    budget_rows = []
    for method in METHODS:
        for q in BUDGETS:
            entry = {'method':method,'budget':q,'actual_area':float(np.mean([r[f'{method}_actual_area{q}'] for r in rows])),
                'preference':float(np.mean([r[f'{method}_preference{q}'] for r in rows])),
                'margin_drop':float(np.mean([r[f'{method}_margin_drop{q}'] for r in rows])),
                'rankflip':float(np.mean([r[f'{method}_rankflip{q}'] for r in rows]))}
            budget_rows.append(entry)
            lines.append(f"| {LABELS[method]} | {q}% | {entry['actual_area']:.2%} | {entry['preference']:.6f} | {entry['margin_drop']:.6f} | {entry['rankflip']:.2%} |")
    write_csv(root/'budget_results.csv',budget_rows)
    lines += ['', '| Comparison | Budget | Neither | First only | Second only | Both | Exact McNemar p | Preference diff / CI | Margin-drop diff / CI |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for name,c in comparisons.items():
        for q in BUDGETS:
            e=c['budgets'][str(q)]; counts=e['outcome_counts']; p=e['preference']; drop=e['margin_drop']
            lines.append(f"| {name} | {q}% | {counts['neither']} | {counts['first_only']} | {counts['second_only']} | {counts['both']} | {e['exact_mcnemar_pvalue']:.6g} | {p['mean_difference']:+.6f} / {interval(p)} | {drop['mean_difference']:+.6f} / {interval(drop)} |")
    lines += ['', 'Exact two-sided McNemar uses a binomial test on the discordant paired outcomes. Full paired preference/margin-drop Wilcoxon statistics, medians, wins and CIs are in `summary.json`.', '',
        '## Technical validation and reproducibility', '',
        f"Three new unit tests validate the official notebook formula, unchanged multi-head attention, inverse crop geometry, region means, and pre-ReLU contrastive linearity. A five-image technical pilot passed before the full run. All 300 examples passed native-score agreement (<1e-6), gradient linearity (<1e-6), repeated-gradient/no-accumulation, finite maps, expected spatial dimensions, and nonzero gradient checks. Maximum native score discrepancy: {max(r['gradient_native_score_error'] for r in rows):.3g}; maximum gradient linearity discrepancy: {max(r['gradient_linearity_error'] for r in rows):.3g}.", '',
        f"Full model state hashes match before and after explanation generation. Independent saved-evidence verification checked maps, mean region scores, deletion rankings and cumulative masks, first crossings, preference/AUC formulas, paired inference, and verbatim SHAP reuse. All {verify['protected_files_unchanged']:,} protected prior source/result/data files stayed unchanged. No SHAP coalition inference was recomputed.", '',
        'Package, PyTorch, CUDA, cuDNN and GPU versions are in `environment.json`; parameters and source hashes in `config.json` and `model_integrity.json`. Per-example `raw.npz` preserves Q/K/V, spatial weights, objective gradients, pre-ReLU/native patch maps, working-image pixel maps, SLIC scores, and matched cumulative masks. Separate JSON files preserve crop geometry, curves and technical checks.', '', '## Runtime', '']
    shap_rows = json.loads((EXP4/'rows.json').read_text())
    costs = {'gradient_shared_explanation_total_seconds':float(sum(r['shared_gradient_explanation_runtime_seconds'] for r in rows)),
        'gradient_shared_explanation_mean_seconds':float(np.mean([r['shared_gradient_explanation_runtime_seconds'] for r in rows])),
        'gradient_shared_evaluation_total_seconds':float(sum(r['shared_gradient_evaluation_runtime_seconds'] for r in rows)),
        'shap_shared_coalition_inference_total_seconds':float(sum(r['shared_inference_runtime_seconds'] for r in shap_rows)),
        'shap_both_regressions_total_seconds':float(sum(r['point_regression_runtime_seconds']+r['cr_regression_runtime_seconds'] for r in shap_rows))}
    save_json(root/'runtime_summary.json',costs)
    lines += [f"Both gradient objectives together: {costs['gradient_shared_explanation_total_seconds']:.2f}s explanation/technical-check time ({costs['gradient_shared_explanation_mean_seconds']:.4f}s per image), plus {costs['gradient_shared_evaluation_total_seconds']:.2f}s shared evaluator cost. Existing SHAP shared coalition inference: {costs['shap_shared_coalition_inference_total_seconds']:.2f}s, plus {costs['shap_both_regressions_total_seconds']:.2f}s for the two target regressions.", '',
        'Gradient timing includes the native-forward comparison, negative-gradient and repeat-gradient validations, and map projection; SHAP inference includes coalition masking/scoring. These are documented workloads, not identical benchmark scopes. Timing excludes model loading, hash snapshots, report generation, and disk serialization. Identical evaluator images are cached across gradient methods and endpoints.', '',
        '## Interpretation and limitations', '', banswer, '', families, '',
        f"Key B: mean difference {b['mean_difference']:+.6f}, relative mean improvement {b['relative_mean_improvement_percent']:.2f}%, Wilcoxon p={b['wilcoxon']['pvalue']:.6g}, paired CI {interval(b)}. Objective-controlled D: first=CR-SHAP, second=Contrastive Grad-ECLIP, mean difference {d['mean_difference']:+.6f}, p={d['wilcoxon']['pvalue']:.6g}, CI {interval(d)}.", '',
        'If both A and B support a positive difference, this supports contrastive preference as a useful principle across these two explanation families under this evaluator. Otherwise the gradient comparison does not establish the same benefit; absence of statistical support is not proof of no effect. A family difference is specific to the chosen published variant and evaluator and cannot prove universal SHAP superiority.', '',
        'The frozen-model multi-head variant differs from the authors’ recommended single-head demo and is labeled throughout. One final layer, coarse 7×7 resolution, positive ReLU support, mean pixel relevance versus SHAP region contributions, inverse-crop zero padding, whole-region budget overshoot, correctly-ranked-only conditioning, semantic overlap among natural false captions, shared candidate captions, and one model/sample seed limit interpretation. Telea removal tests model preference under perturbations rather than causal faithfulness. No outcome-based tuning, resampling or evaluator changes occurred.', '', '## Failure cases and diagnostic examples', '']
    failure = {'zero_relu_maps':verify['zero_relu_maps'],'largest_gradient_worsenings':[]}
    ordered = sorted(range(300),key=lambda i:(rows[i]['gradient_objective_difference'],rows[i]['image_id']))
    for i in ordered[:5]:
        failure['largest_gradient_worsenings'].append({'image_id':rows[i]['image_id'],'difference':rows[i]['gradient_objective_difference'],
            'positive_caption':rows[i]['positive_caption'],'negative_caption':rows[i]['negative_caption']})
    save_json(root/'failure_cases.json',failure)
    lines += [f"Zero native ReLU maps: Pointwise {len(verify['zero_relu_maps']['grad_point'])}, Contrastive {len(verify['zero_relu_maps']['grad_cr'])}. Finite zero maps in the full sample are preserved and evaluated with deterministic region-index ties, never replaced. Largest objective worsenings and their captions are in `failure_cases.json`.", '',
        'Fifteen figures show the five largest gradient objective improvements, five closest-to-zero unused differences, and five smallest differences. These outcome-stratified diagnostic examples are not representative estimates of population performance. Selection and exact IDs are saved.', '']
    improved = list(reversed(ordered[-5:])); worsened=ordered[:5]; used=set(improved+worsened)
    neutral = sorted((i for i in range(300) if i not in used),key=lambda i:(abs(rows[i]['gradient_objective_difference']),rows[i]['image_id']))[:5]
    choices=[('improvement',i) for i in improved]+[('near_zero',i) for i in neutral]+[('worsening',i) for i in worsened]
    save_json(root/'figure_selection.json',[{'stratum':s,'index':i,'image_id':rows[i]['image_id']} for s,i in choices])
    figs=root/'figures'; figs.mkdir()
    for stratum,i in choices:
        row=rows[i]; source=EXP4/f'example_{i:03d}'; folder=root/f'example_{i:03d}'
        image=np.asarray(Image.open(source/'original.png')); old=np.load(source/'raw.npz'); raw=np.load(folder/'raw.npz')
        matched=json.loads((folder/'matched_deletion.json').read_text()); old_matched=json.loads((source/'matched_deletion.json').read_text())
        fig,axes=plt.subplots(2,4,figsize=(20,10)); axes=axes.ravel(); fig.subplots_adjust(top=.75,hspace=.3,wspace=.25)
        axes[0].imshow(image); axes[0].set_title('Original'); axes[1].imshow(mark_boundaries(image,raw['segments'])); axes[1].set_title('Frozen SLIC')
        limit=max(np.max(abs(old['phi_pointwise'])),np.max(abs(old['phi_cr_shap'])),1e-10)
        for ax,key,title in [(axes[2],'phi_pointwise','Pointwise SHAP'),(axes[3],'phi_cr_shap','CR-SHAP')]:
            ax.imshow(image); ax.imshow(old[key][raw['segments']],cmap='coolwarm',vmin=-limit,vmax=limit,alpha=.65); ax.set_title(title)
        for ax,method in [(axes[4],'grad_point'),(axes[5],'grad_cr')]:
            pixels=raw[method+'_pixel_map']; ax.imshow(image); ax.imshow(pixels,cmap='inferno',vmin=0,vmax=max(pixels.max(),1e-10),alpha=.65); ax.set_title(LABELS[method])
        for method in ('grad_point','grad_cr'):
            c=matched[method]; axes[6].plot(c['auc_fractions'],c['auc_preferences'],'o-',label=f"{LABELS[method]} {c['matched_pdauc50']:.3f}")
        for method in METHODS:
            c=matched[method] if method.startswith('grad') else old_matched['point' if method=='shap_point' else 'cr']
            axes[7].plot(c['auc_fractions'],c['auc_preferences'],'o-',label=LABELS[method])
        for ax in axes[:6]: ax.axis('off')
        for ax,title in [(axes[6],'Gradient objective comparison'),(axes[7],'Complete 2x2 comparison')]:
            ax.set(xlabel='Actual removed area',ylabel='Positive preference',ylim=(0,1),title=title); ax.axhline(.5,color='gray',linestyle=':'); ax.legend(fontsize=7)
        fig.suptitle('\n'.join([f"{row['image_id']} | {stratum} | gradient objective difference {row['gradient_objective_difference']:+.6f}",
            'T+: '+textwrap.fill(row['positive_caption'],150),'T-: '+textwrap.fill(row['negative_caption'],150),
            f"original cos+={row['original_positive_cosine']:.6f}, cos-={row['original_negative_cosine']:.6f}, margin={row['original_margin']:.6f}"]),fontsize=11,y=.99)
        name=f'{stratum}_{i:03d}.png'; fig.savefig(figs/name,dpi=120,bbox_inches='tight'); plt.close(fig); old.close(); raw.close()
        lines += [f'![{stratum}: image {row["image_id"]}](figures/{name})','']
    lines += ['Experiment 5 is complete. No Experiment 6 or other follow-up was launched. Human review is required before further work.','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Complete Experiment5 report, tables, failure evidence, runtime and 15 figures generated; STOP',flush=True)


if __name__=='__main__': main()
