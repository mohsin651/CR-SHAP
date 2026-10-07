"""Render the frozen external-validation results and 15 diagnostic figures."""
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
from .core import BUDGETS


def main():
    root = Path('results/experiment_4')
    if (root/'report.md').exists():
        raise FileExistsError(root/'report.md')
    verification = json.loads((root/'verification.json').read_text())
    if not verification['all_passed'] or verification['n'] != 300:
        raise ValueError('Complete independently verified run required')
    rows = json.loads((root/'rows.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    screen = json.loads((root/'screening_summary.json').read_text())
    selection = json.loads((root/'selection_report.json').read_text())
    primary = summary['primary']
    supported = primary['mean_difference'] > 0 and primary['wilcoxon']['pvalue'] < .05 and primary['paired_bootstrap_ci95']['low'] > 0
    lines = ['# Experiment 4: external validation of CR-SHAP on natural Flickr30k retrieval', '',
        'The primary matched-area endpoint supports a CR-SHAP advantage in this sample.' if supported else
        'The primary matched-area endpoint does not provide clear statistical support for a CR-SHAP advantage in this sample.', '',
        '## Dataset, retrieval, and frozen sample', '',
        'Original Karpathy TEST annotations: 1,000 unique images, 5,000 captions, exactly five per image. Image-caption ownership and an independent retrieval annotation mirror were validated. Captions are used verbatim from the original annotations.', '',
        f"Full retrieval screening: {screen['correctly_ranked']} strictly correctly ranked images ({screen['correctly_ranked_percent']:.2f}%), {screen['incorrect_or_tied']} incorrect or tied, {screen['ties']} exact margin ties.", '',
        '| Direction | R@1 | R@5 | R@10 |', '|---|---:|---:|---:|']
    for direction, metrics in screen['retrieval'].items():
        if isinstance(metrics, dict):
            lines.append(f"| {direction} | " + ' | '.join(f'{metrics[f"R@{k}"]:.2%}' for k in (1,5,10)) + ' |')
    lines += ['', 'Retrieval metrics describe the original zero-shot CLIP model. Post-hoc CR-SHAP does not change retrieval accuracy.', '',
        '| Screening quantity | Mean | Median | Std (population) | Minimum | Maximum |', '|---|---:|---:|---:|---:|---:|']
    for key, values in screen['distributions'].items():
        lines.append(f'| {key} | ' + ' | '.join(f'{values[k]:.6f}' for k in ('mean','median','standard_deviation','minimum','maximum')) + ' |')
    lines += ['', f"Uniform seed-2026 sampling selected 300 unique correctly ranked query images. All initial images were checked for coalition feasibility before explanation outcomes; {selection['technical_exclusions']} technical exclusions and {selection['reserve_images_checked']} reserve checks are documented in `technical_exclusions.json`. Replacements followed the pre-generated seeded reserve list.", '',
        'T+ is the highest-scoring ground-truth caption. T- is the highest-scoring caption owned by another test image, selected from the complete 5,000-caption pool. Caption IDs, ownership, scores, and all screening rows are saved in `data/flickr30k_screening_exp4/screening.csv`.', '',
        'Frozen CLIP `openai/clip-vit-base-patch32`, commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; eval mode, no gradients. EXIF transpose, RGB, aspect-preserving maximum side 512, then the unchanged CLIP processor/tokenizer. SLIC: 16 requested regions, compactness 30, start label 0, channel axis -1, scikit-image 0.24.0 defaults. Kernel SHAP: unchanged constrained solver, 128 unique shared coalitions, seed 42, global RGB mean masking. One normalized image embedding per coalition supplies both raw cosine scores.', '',
        '## Paired endpoints', '',
        '| Endpoint | Point mean / median | CR mean / median | Mean / median difference | Relative improvement | CR / Point wins / ties | Wilcoxon statistic / p | Paired 95% CI |',
        '|---|---|---|---|---:|---|---|---|']
    for label, s in [('Primary: MatchedArea-PDAUC-50',primary), ('Secondary: PositiveOnly-PDAUC',summary['secondary_positive_only'])]:
        ci = s['paired_bootstrap_ci95']
        lines.append(f"| {label} | {s['point_mean']:.6f} / {s['point_median']:.6f} | {s['cr_mean']:.6f} / {s['cr_median']:.6f} | {s['mean_difference']:+.6f} / {s['median_difference']:+.6f} | {s['relative_mean_improvement_percent']:.2f}% | {s['cr_wins']} / {s['point_wins']} / {s['ties']} | {s['wilcoxon']['statistic']:.6g} / {s['wilcoxon']['pvalue']:.6g} | [{ci['low']:+.6f}, {ci['high']:+.6f}] |")
    lines += ['', 'n=300. Paired two-sided Wilcoxon tests and 10,000 paired percentile bootstrap resamples, seed 42, over unique query images. No image-cluster correction is needed for repeated query images because each occurs once. Positive AUC/preference differences mean Pointwise minus CR; positive margin-drop and rank-flip differences mean CR minus Pointwise.', '',
        'Both evaluations use Telea radius 3 on the original image with each cumulative whole-region mask; the validated full-mask global-mean fallback is unchanged. Primary ranking uses all signed SHAP values descending, index ties ascending. Each 10/20/30/40/50% budget uses the first whole-region crossing. AUC integrates the original state and five crossing states against actual areas and divides by the final area; repeated crossings have zero-width intervals, with no tail. Secondary deletion removes strictly positive regions only, extending the final preference constantly to 100% solely for integration. Preference is sigmoid(gamma times cosine margin), gamma approximately 100.', '',
        '## Secondary fixed budgets', '',
        '| Budget | Actual area Point / CR | Preference Point / CR | Margin drop Point / CR | Flip Point / CR | Neither / Point only / CR only / Both | Exact McNemar p |',
        '|---|---|---|---|---|---|---:|']
    budget_rows = []
    for q in BUDGETS:
        b = summary['budgets'][str(q)]
        p, d, c = b['preference'], b['margin_drop'], b['rankflip_2x2']
        lines.append(f"| {q}% | {b['point_mean_actual_area']:.2%} / {b['cr_mean_actual_area']:.2%} | {p['point_mean']:.6f} / {p['cr_mean']:.6f} | {d['point_mean']:.6f} / {d['cr_mean']:.6f} | {b['point_rankflip_rate']:.2%} / {b['cr_rankflip_rate']:.2%} | {c['neither_flipped']} / {c['point_only_flipped']} / {c['cr_only_flipped']} / {c['both_flipped']} | {b['exact_mcnemar_pvalue']:.6g} |")
        budget_rows.append({'budget': q, **{k:v for k,v in b.items() if not isinstance(v,dict)}, **c})
    write_csv(root/'budget_results.csv', budget_rows)
    lines += ['', 'Exact two-sided McNemar uses a binomial test on discordant pairs. Budget tests are secondary and unadjusted. Complete preference/margin-drop statistics and CIs are saved in `summary.json`.', '', '## Exploratory difficulty relationships', '']
    for key, c in summary['exploratory'].items():
        lines.append(f"- {key} versus matched PDAUC difference: Spearman rho={c['spearman_rho']}, p={c['pvalue']}.")
    lines += ['', 'These correlations are exploratory and do not redefine the primary hypothesis.', '', '## Evidence and interpretation', '',
        f"Independent verification reconstructed all 1,000 retrieval winners, recall metrics, reserve sampling, {verification['shared_coalitions_validated']:,} coalition masks/perturbations, shared cosine scores, constrained regressions, signed rankings, budget masks, formulas, and paired statistics. Maximum linearity error: {verification['max_abs_linearity_error']:.3g}. All {verification['protected_files_unchanged']:,} protected prior files remained unchanged. SugarCrepe was not rerun.", '',
        'This measures external validity under the frozen CLIP/Telea evaluator. Other-image ownership does not ensure that a retrieved caption is semantically false: natural caption overlap is possible. Conditioning on correctly ranked images, technical exclusions, approximate SHAP, whole-region overshoot, shared candidate captions, one model and sampling seed limit generalization. This does not establish causal faithfulness. The primary endpoint stays matched-area regardless of the secondary result.', '',
        'Per-example original images, segmentations, coalitions, normalized image/text embeddings, both cosine caches, all three SHAP vectors, hashes, deletion masks and curves are preserved in `example_000` through `example_299`. Timing separates shared coalition inference, method regressions, and evaluator cost. Fifteen outcome-stratified figures are diagnostic; selection is deterministic and not representative.', '']
    figures = root/'figures'
    figures.mkdir()
    by_difference = sorted(range(len(rows)), key=lambda i:(rows[i]['matched_pdauc_difference'],rows[i]['image_id']))
    improvements = list(reversed(by_difference[-5:]))
    worsening = by_difference[:5]
    used = set(improvements+worsening)
    neutral = sorted((i for i in range(len(rows)) if i not in used), key=lambda i:(abs(rows[i]['matched_pdauc_difference']),rows[i]['image_id']))[:5]
    choices = [('improvement',i) for i in improvements]+[('near_zero',i) for i in neutral]+[('worsening',i) for i in worsening]
    save_json(root/'figure_selection.json', {'rule': 'Five largest differences, five closest to zero among remaining images, five smallest differences; deterministic image-ID ties.',
        'examples': [{'stratum':s,'index':i,'image_id':rows[i]['image_id'],'difference':rows[i]['matched_pdauc_difference']} for s,i in choices]})
    for stratum, i in choices:
        row = rows[i]
        folder = root/f'example_{i:03d}'
        image = np.asarray(Image.open(folder/'original.png'))
        raw = np.load(folder/'raw.npz')
        matched = json.loads((folder/'matched_deletion.json').read_text())
        positive = json.loads((folder/'positive_deletion.json').read_text())
        fig, axes = plt.subplots(2,3,figsize=(16,10))
        axes = axes.ravel()
        fig.subplots_adjust(top=.76,hspace=.35,wspace=.3)
        axes[0].imshow(image); axes[0].set_title('Original')
        axes[1].imshow(mark_boundaries(image,raw['segments'])); axes[1].set_title('Frozen SLIC')
        limit = max(np.max(abs(raw['phi_pointwise'])),np.max(abs(raw['phi_cr_shap'])),1e-10)
        for ax,key,title in [(axes[2],'phi_pointwise','Pointwise SHAP'),(axes[3],'phi_cr_shap','CR-SHAP')]:
            ax.imshow(image)
            heat = ax.imshow(raw[key][raw['segments']],cmap='coolwarm',vmin=-limit,vmax=limit,alpha=.65)
            ax.set_title(title)
        fig.colorbar(heat,ax=axes[2:4].tolist(),shrink=.55,label='Signed raw cosine SHAP')
        for prefix,label in [('point','Pointwise'),('cr','CR-SHAP')]:
            c = matched[prefix]
            axes[4].plot(c['auc_fractions'],c['auc_preferences'],'o-',label=f"{label}: {c['matched_pdauc50']:.4f}")
            c = positive[prefix]
            axes[5].plot(c['fractions'],c['preferences'],'o-',label=f"{label}: {c['auc']:.4f}")
            if c['fractions'][-1]<1:
                axes[5].plot([c['fractions'][-1],1],[c['preferences'][-1]]*2,'--')
        for ax,title in [(axes[4],'Primary matched-area deletion'),(axes[5],'Secondary positive-only; dashed integration tails')]:
            ax.set(xlabel='Actual removed pixel fraction',ylabel='Positive preference',ylim=(0,1),title=title)
            ax.axhline(.5,color='gray',linestyle=':'); ax.legend(fontsize=8)
        for ax in axes[:4]: ax.axis('off')
        fig.suptitle('\n'.join([f"Image {row['image_id']} | {stratum} | matched difference {row['matched_pdauc_difference']:+.6f}",
            'T+: '+textwrap.fill(row['positive_caption'],120), 'Strongest false T-: '+textwrap.fill(row['negative_caption'],120),
            f"cos+={row['original_positive_cosine']:.6f}; cos-={row['original_negative_cosine']:.6f}; margin={row['original_margin']:.6f}"]),fontsize=11,y=.99)
        name = f'{stratum}_{i:03d}.png'
        fig.savefig(figures/name,dpi=130,bbox_inches='tight'); plt.close(fig); raw.close()
        lines.append(f'![{stratum}: image {row["image_id"]}](figures/{name})')
        lines.append('')
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Report and 15 diagnostic figures completed',flush=True)


if __name__ == '__main__':
    main()
