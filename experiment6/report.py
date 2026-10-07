"""Report only this grounding experiment and then stop."""
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiment2.io import save_json
from .prepare import EXP4,ROOT
from .run import overlay


def ci(s,scale=1):
    c=s['paired_bootstrap_ci95']; return f"[{scale*c['low']:+.4f}, {scale*c['high']:+.4f}]"


def main():
    if (ROOT/'report.md').exists(): raise FileExistsError(ROOT/'report.md')
    v=json.loads((ROOT/'verification.json').read_text())
    if not v['all_passed']: raise ValueError('Verified results required')
    rows=json.loads((ROOT/'rows.json').read_text()); s=json.loads((ROOT/'summary.json').read_text())
    sample=json.loads((ROOT/'sample_summary.json').read_text()); manifest=json.loads((ROOT/'matched_manifest.json').read_text())
    pg,eib,area=s['pointing_game'],s['eib'],s['positive_area']
    def direction(stat):
        c=stat['paired_bootstrap_ci95']
        return 'higher' if c['low']>0 else 'lower' if c['high']<0 else 'uncertain'
    pg_direction=direction(pg); energy_direction=direction(eib)
    q1=f"Pointing Game is {pg_direction} for CR-SHAP (exact McNemar p={pg['exact_mcnemar_pvalue']:.6g}); EIB is {energy_direction} (Wilcoxon p={eib['wilcoxon']['pvalue']:.6g}). A nonsignificant difference does not establish equivalence."
    q2=f"CR-SHAP uses {'less' if area['mean_difference']<0 else 'more'} positive image area: {area['point_mean']:.2%} versus {area['cr_mean']:.2%}, paired difference {100*area['mean_difference']:+.2f} percentage points, CI {ci(area,100)}. Area is contextual, not grounding quality."
    q3=('The grounding results indicate a tradeoff on at least one spatial metric, despite the previously observed retrieval-deletion advantage.' if pg_direction=='lower' or energy_direction=='lower' else
        'This sample does not demonstrate a grounding cost of the previously observed retrieval-deletion advantage. It does not prove absence of a cost or statistical equivalence.')
    lines=['# Experiment 6: Flickr30k Entities grounding of frozen SHAP explanations','',
        '## Scientific question','',
        'Do the explanations that better explain relative CLIP retrieval preference also localize human-annotated visual entities in T+? This experiment measures grounding, not deletion faithfulness, using only Pointwise SHAP and CR-SHAP. No model inference or SHAP recomputation occurred.','',
        '## Frozen annotation sample and targets','',
        f"Original frozen sample: **300**. Successfully caption-matched: **{sample['successfully_caption_matched']}**. Grounding-evaluable: **{sample['grounding_evaluable']}**. Excluded for annotation/technical reasons: **{sample['excluded']}**. All excluded IDs and reasons are in `exclusions.json`; no replacement or outcome-based filtering occurred.",'',
        'Annotations come from the [official Flickr30k Entities repository](https://github.com/BryanPlummer/flickr30k_entities), pinned at commit `68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b`. Its original sentence/XML parser is archived with the annotation files. The original Karpathy caption ID determines the sentence index; tokens must match exactly after HTML unescape, case folding, and punctuation whitespace normalization. Words and punctuation are not semantically substituted. Original caption text, IDs, index and entity sentence are retained.','',
        'For each T+, all explicit valid boxes of all linked visual phrases are included. Multiple boxes per phrase are retained, duplicate boxes are deduplicated, and overlaps contribute once in a union mask. Nonvisual/id0 phrases and phrases without explicit boxes contribute no target. Scene-type phrases with explicit boxes are included; scene/no-box flags alone do not create a whole-image box. Invalid geometry causes a documented technical exclusion. Targets and rules were saved before attributions were scored.','',
        '## Coordinates and attribution representation','',
        'Official parser coordinates are zero-based inclusive. Maxima are increased by one to obtain half-open pixel edges, clipped to the raw image dimensions, transformed through any EXIF orientation, and scaled by the exact saved working/oriented dimension ratios. Pixel-center inclusion uses ceil(edge − 0.5) raster bounds. No CLIP center crop is applied to ground truth: both boxes and SLIC are evaluated on the saved working image. Original/transformed boxes, dimensions, clipping and orientation are recorded per example.','',
        f"Nonidentity EXIF cases: {sample['nonidentity_exif_examples']}; raw-to-working resized cases: {sample['resized_examples']}. A fixed technical pilot checked image identity, box alignment, SLIC, conserved energy, pointing and area; overlays were visually reviewed before full statistical analysis.",'',
        'Saved SHAP values are contributions of whole SLIC regions. The positive spatial density for a pixel in region j is max(phi_j,0)/n_j, where n_j is its pixel count. This preserves each region’s total positive contribution and prevents region area from accidentally multiplying energy. Both metrics use this same map for both methods. EIB numerator is density summed inside the GT union; denominator is total positive SHAP mass. Numerator, denominator and ratio are saved.','',
        'Pointing Game chooses the greatest positive density region, with the lowest region index for equal densities, then the pixel nearest that region’s centroid, with row-major y/x ties. The location is always within the selected region. This is a region-aware tie rule for the plateau, rather than an arbitrary first boundary pixel. It ranks contribution density, not the total-contribution ranking used for prior deletion experiments. Zero-positive-energy cases remain in the sample as Pointing failures and EIB=0, with explicit flags.','',
        '## Headline paired results','',
        '| Metric | Pointwise SHAP | CR-SHAP | CR − Pointwise | Paired 95% CI | p |',
        '|---|---:|---:|---:|---|---:|',
        f"| Primary: Pointing Game ↑ | {pg['point_mean']:.2%} | {pg['cr_mean']:.2%} | {100*pg['mean_difference']:+.2f} pp | {ci(pg,100)} pp | {pg['exact_mcnemar_pvalue']:.6g} |",
        f"| Secondary: EIB ↑ | {eib['point_mean']:.6f} | {eib['cr_mean']:.6f} | {eib['mean_difference']:+.6f} | {ci(eib)} | {eib['wilcoxon']['pvalue']:.6g} |",
        f"| Context: positive attribution area ↓ | {area['point_mean']:.2%} | {area['cr_mean']:.2%} | {100*area['mean_difference']:+.2f} pp | {ci(area,100)} pp | {area['wilcoxon']['pvalue']:.6g} |",'',
        f"Paired n={s['n']}. All difference CIs use 10,000 paired percentile bootstrap resamples over unique images, seed 42. Exact two-sided McNemar on discordant pointing outcomes is the primary binary test; secondary EIB uses paired two-sided Wilcoxon. Tests are unadjusted. The area p-value describes selectivity only.",'',
        f"Pointing paired counts: {pg['paired_outcomes']}. Mean GT union coverage: {s['gt_union_area_mean']:.2%}; broad unions can make localization permissive.",'',
        f"EIB medians: Pointwise {eib['point_median']:.6f}, CR {eib['cr_median']:.6f}; median paired difference {eib['median_difference']:+.6f}; CR wins {eib['cr_wins']}, Pointwise wins {eib['point_wins']}, ties {eib['ties']}; Wilcoxon statistic {eib['wilcoxon']['statistic']:.6g}.",'',
        f"Positive area medians: Pointwise {area['point_median']:.2%}, CR {area['cr_median']:.2%}; median paired difference {100*area['median_difference']:+.2f} pp. Zero-positive-energy cases: Pointwise {sum(r['point_zero_positive_energy'] for r in rows)}, CR {sum(r['cr_zero_positive_energy'] for r in rows)}.",'',
        '## Conclusions','',f'**1. Localization:** {q1}','',f'**2. Selectivity:** {q2}','',f'**3. Faithfulness versus grounding:** {q3}','',
        'Experiments 3–5 tested how perturbations change CLIP retrieval preference. Human box overlap measures a distinct property; improved deletion AUC is not proof of better entity grounding. Bounding-box overlap does not establish causal localization or human interpretability. Contrastive evidence can lie outside the positive-caption entities, but these results alone do not explain why.','',
        '## Validation, limitations and reproducibility','',
        f"All {v['n']} evaluable cases passed independent saved-evidence checks: exact reused SHAP/SLIC, GT raster, conserved density, continuous-box point membership, separate region-overlap energy formula, positive area and paired statistics. All {v['protected_files_unchanged']:,} protected prior files remained unchanged. Unit tests cover token identity, scaling/inclusive endpoints, clipping, EXIF orientation, region energy conservation, centroid ties and zero-energy handling.",'',
        'The existing 300 correctly ranked retrieval examples are not the whole Flickr30k population. Union boxes may include large backgrounds/scene entities and do not provide pixel-accurate segmentation. Coarse SLIC plateaus require a deterministic representative point; conserved density changes the peak from the greatest whole-region contribution in some examples. Positive-only mass ignores negative evidence. One sample/model/segmentation, annotation availability, and shared candidate captions limit generalization. No new aggregation, box subset or normalization was chosen using outcomes.','',
        'Artifacts: matched/excluded manifests, frozen config/source hashes, original/transformed phrase boxes and union masks in `targets/`, density maps and reused phi/SLIC in `examples/`, per-example results, statistics, environment, geometry pilot/review, verification and deterministic figures. Dataset/parser hashes and references to each reused Experiment 4 artifact are retained.','',
        '## Diagnostic examples','',
        'For each pointing outcome stratum, choose the first two image IDs lexicographically. Additionally show the three largest EIB worsenings for CR among examples with positive CR energy outside the GT union, ties by image ID. These illustrate possible outside-box discriminative evidence, without establishing its semantics or explaining the retrieval mechanism. Outcome-based figure selection is diagnostic, not quantitative evidence.','']
    categories={'both_grounded':lambda r:r['point_pointing_game'] and r['cr_pointing_game'],
        'cr_only_grounded':lambda r:not r['point_pointing_game'] and r['cr_pointing_game'],
        'point_only_grounded':lambda r:r['point_pointing_game'] and not r['cr_pointing_game'],
        'neither_grounded':lambda r:not r['point_pointing_game'] and not r['cr_pointing_game']}
    choices=[]
    for name,predicate in categories.items():
        choices.extend((name,r) for r in sorted((r for r in rows if predicate(r)),key=lambda r:r['image_id'])[:2])
    choices.extend(('outside_box_evidence',r) for r in sorted((r for r in rows if r['cr_eib_denominator']>0 and r['cr_eib']<1),
        key=lambda r:(r['cr_eib']-r['point_eib'],r['image_id']))[:3])
    save_json(ROOT/'figure_selection.json',[{'stratum':name,'image_id':r['image_id'],'experiment4_index':r['experiment4_index']} for name,r in choices])
    figs=ROOT/'figures'; figs.mkdir(); mappings={r['image_id']:r for r in manifest}
    for name,row in choices:
        i=row['experiment4_index']; record=mappings[row['image_id']]
        image=np.asarray(Image.open(EXP4/f'example_{i:03d}'/'original.png'))
        raw=np.load(ROOT/'examples'/f'{i:03d}'/'grounding.npz')
        fig,axes=plt.subplots(1,3,figsize=(16,6)); fig.subplots_adjust(top=.70)
        overlay(axes[0],image,record,raw['gt_union']); axes[0].set_title('All valid T+ boxes and union')
        limit=max(raw['point_positive_density'].max(),raw['cr_positive_density'].max(),1e-12)
        for ax,prefix,title in [(axes[1],'point','Pointwise SHAP'),(axes[2],'cr','CR-SHAP')]:
            overlay(ax,image,record)
            ax.imshow(raw[prefix+'_positive_density'],cmap='inferno',vmin=0,vmax=limit,alpha=.65,extent=(0,image.shape[1],image.shape[0],0))
            if row[prefix+'_point_xy'] is not None:
                x,y=row[prefix+'_point_xy']; ax.plot(x+.5,y+.5,'c+',markersize=14,markeredgewidth=2)
            ax.set_title(f"{title}: PG={int(row[prefix+'_pointing_game'])}, EIB={row[prefix+'_eib']:.3f}, positive area={row[prefix+'_positive_area_fraction']:.1%}")
        fig.suptitle('\n'.join([f"{row['image_id']} | {name}",'T+: '+textwrap.fill(row['positive_caption'],135),'T-: '+textwrap.fill(row['negative_caption'],135)]),fontsize=11,y=.99)
        filename=f'{name}_{i:03d}.png'; fig.savefig(figs/filename,dpi=130,bbox_inches='tight'); plt.close(fig); raw.close()
        lines += [f'![{name}: {row["image_id"]}](figures/{filename})','']
    lines += ['Experiment 6 is complete. No Grad-ECLIP grounding, further dataset, model, or Experiment 7 was run. Stopped for human review.','']
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Experiment6 report and grounding figures complete; STOP',flush=True)


if __name__=='__main__': main()
