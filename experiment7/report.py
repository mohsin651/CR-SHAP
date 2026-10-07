"""Final planned backbone replication report; no subsequent experiment launch."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment2.io import save_json,write_csv
from experiment3.core import BUDGETS
from .model import COMMIT,MODEL
from .screen import root_for

ROOT=Path('results/experiment_7')


def interval(s):
    c=s['paired_bootstrap_ci95']; return f"[{c['low']:+.6f}, {c['high']:+.6f}]"


def main():
    if (ROOT/'report.md').exists(): raise FileExistsError(ROOT/'report.md')
    summaries={}; screens={}; selections={}; rows={}; validations={}
    for arm in ('sugarcrepe','flickr30k'):
        path=ROOT/arm/'full'
        validations[arm]=json.loads((path/'verification.json').read_text())
        if not validations[arm]['all_passed'] or validations[arm]['n']!=200: raise ValueError('Exactly200 verified per arm required')
        summaries[arm]=json.loads((path/'summary.json').read_text()); screens[arm]=json.loads((path/'screening_summary.json').read_text())
        selections[arm]=json.loads((path/'selection_report.json').read_text()); rows[arm]=json.loads((path/'rows.json').read_text())
    positive=[s['primary']['mean_difference']>0 for s in summaries.values()]
    supported=[s['primary']['paired_bootstrap_ci95']['low']>0 for s in summaries.values()]
    if all(positive) and all(supported):
        conclusion='Both datasets replicate the positive CR-SHAP direction with paired confidence intervals above zero. The advantage is not specific to ViT-B/32 in these two settings.'
    elif all(positive):
        conclusion='Both datasets show directionally consistent positive effects under ViT-B/16, but at least one interval includes zero; statistical support is weaker in that setting.'
    elif any(positive):
        conclusion='The two datasets give mixed directional evidence under ViT-B/16. Generalization is model/dataset-dependent and requires caution.'
    else:
        conclusion='Neither dataset reproduces the positive direction under ViT-B/16. The earlier advantage may depend materially on ViT-B/32; no method settings or sample sizes were changed.'
    lines=['# Experiment 7: final CLIP ViT-B/16 backbone replication','',conclusion,'',
        '## Scientific question and frozen design','',
        'Does Pointwise-to-CR SHAP improvement persist under a second CLIP backbone on controlled SugarCrepe pairs and natural Flickr30k retrieval? The arms are analyzed separately and never pooled. Each explains exactly 200 examples from a fresh B/16-eligible pool. Experiments 1–6 were not rerun, modified, or optimized.','',
        f"Model: `{MODEL}`, pinned revision `{COMMIT}`. All parameters are frozen, eval mode, with no gradients/optimizer/training. Full state hashes before/after screening and each pilot/full explanation run agree. Package, torch/CUDA and GPU details are saved per stage; weights and processor are pinned to the same revision.",'',
        'The unchanged working-image preparation is EXIF transpose, RGB, aspect-preserving max-side512 followed by CLIP processor resize/center crop. B/16 has 224×224 input, 14×14 transformer patches, and 197 tokens including CLS. These are not the SHAP features: SLIC still requests16 regions, compactness30, start_label0, channel_axis−1, scikit-image0.24.0 defaults. No methodological parameter needed to change for the patch geometry.','',
        'Both arms use128 unique coalitions, seed42, mandatory empty/full/singletons/complements, the unchanged constrained Kernel SHAP solver, and original global-RGB mean fill. One normalized image embedding per coalition supplies both caption cosines. Raw positive cosine is Pointwise; positive-minus-negative raw cosine is CR. Negative-caption regression validates Shapley linearity. Coalitions, embeddings, scores, all phi vectors, masks/hashes and fit diagnostics are preserved.','',
        'Primary: MatchedArea-PDAUC-50, signed ranking over all SLIC regions, index ties ascending, whole-region first crossings of10/20/30/40/50%, actual pixel fractions, Telea radius3 on the original image, unchanged full-mask mean fallback, original-state inclusion, normalized six-state trapezoidal area, no extrapolation. Preference uses sigmoid(gamma × margin) with each model’s pretrained logit scale. PositiveOnly-PDAUC remains secondary, deleting strictly positive regions and extending the terminal preference to100% for integration only.','',
        '## Screening and sample construction','',
        'Selection seed2027 was fixed before outcomes. Each arm uniformly permutes eligible unique image IDs and uniformly chooses one eligible instance per image. This meets the200 unique-example target while avoiding repeated-image dependence. SugarCrepe is not category-balanced: it samples images from the complete B/16-eligible pool rather than replicating the old equal-category weighting. This sampling difference limits direct effect-size comparisons. All initial200 segmentations are checked before explanations; infeasible examples are replaced only from the frozen reserve order, never from outcomes.','']
    for arm in ('sugarcrepe','flickr30k'):
        screen=screens[arm]; selection=selections[arm]
        lines += [f'### {arm}','',f"Complete B/16 screening: {screen['screened']} instances, {screen['unique_images_screened']} unique images; {screen['correctly_ranked']} strictly correct, {screen['incorrect']} incorrect, {screen['ties']} ties. Selection:200 unique images, {selection['eligible_unique_images']} eligible images, {selection['technical_exclusions']} technical exclusions. Gamma={screen['gamma']:.6f}.",'']
        if arm=='sugarcrepe':
            lines += ['Positive/negative captions are the existing official SugarCrepe pair. All7511 official instances were freshly screened using B/16; B/32 correctness was not inherited.', '',f"Selected category counts: {selection['categories']}.",'']
        else:
            lines += ['The standard Karpathy test set retains1000 images and5000 owned captions. New normalized B/16 embeddings and the full similarity matrix determine the highest-scoring ground-truth T+ and strongest other-image T− independently for every image. B/32 caption competitors/embeddings are not reused.', '',
                '| Direction | R@1 | R@5 | R@10 |','|---|---:|---:|---:|']
            for direction in ('image_to_text','text_to_image'):
                metrics=screen['retrieval'][direction]
                lines.append(f'| {direction} | '+' | '.join(f'{metrics[f"R@{k}"]:.2%}' for k in (1,5,10))+' |')
            lines += ['',f"Original-margin distribution: {screen['distributions']['original_margin']}.",'',
                'Retrieval is an original-model sanity check, not an improvement caused by post-hoc SHAP.','']
    lines += ['## Headline primary results','',
        '| Dataset | Pointwise mean / median | CR mean / median | Mean / median difference | Relative improvement | CR / Point wins / ties | Paired95% CI | Wilcoxon statistic / p |',
        '|---|---|---|---|---:|---|---|---|']
    primary_rows=[]
    for arm,s in summaries.items():
        a=s['primary']
        lines.append(f"| {arm} | {a['point_mean']:.6f} / {a['point_median']:.6f} | {a['cr_mean']:.6f} / {a['cr_median']:.6f} | {a['mean_difference']:+.6f} / {a['median_difference']:+.6f} | {a['relative_mean_improvement_percent']:.2f}% | {a['cr_wins']} / {a['point_wins']} / {a['ties']} | {interval(a)} | {a['wilcoxon']['statistic']:.6g} / {a['wilcoxon']['pvalue']:.6g} |")
        primary_rows.append({'dataset':arm,**{k:v for k,v in a.items() if not isinstance(v,dict)},'ci_low':a['paired_bootstrap_ci95']['low'],'ci_high':a['paired_bootstrap_ci95']['high'],'wilcoxon_p':a['wilcoxon']['pvalue']})
    write_csv(ROOT/'primary_results.csv',primary_rows)
    lines += ['', 'n=200 paired unique images in each arm. Positive AUC/preference differences mean Pointwise minus CR; positive margin-drop differences mean CR minus Pointwise. Tests are paired two-sided Wilcoxon and paired percentile bootstrap10000 resamples, seed42. No cross-dataset pooled p-value or cross-backbone effect-size significance test was performed. Secondary budgets are unadjusted.','',
        '## Secondary PositiveOnly-PDAUC','',
        '| Dataset | Point mean / median | CR mean / median | Mean / median difference | Relative improvement | CR / Point wins / ties | 95% CI | Wilcoxon p |',
        '|---|---|---|---|---:|---|---|---:|']
    for arm,s in summaries.items():
        a=s['secondary_positive_only']
        lines.append(f"| {arm} | {a['point_mean']:.6f} / {a['point_median']:.6f} | {a['cr_mean']:.6f} / {a['cr_median']:.6f} | {a['mean_difference']:+.6f} / {a['median_difference']:+.6f} | {a['relative_mean_improvement_percent']:.2f}% | {a['cr_wins']} / {a['point_wins']} / {a['ties']} | {interval(a)} | {a['wilcoxon']['pvalue']:.6g} |")
    lines += ['', '## Secondary fixed budgets','',
        '| Dataset | Budget | Actual area Point / CR | Preference Point / CR | Margin drop Point / CR | RankFlip Point / CR | Neither / Point only / CR only / Both | Exact McNemar p |',
        '|---|---:|---|---|---|---|---|---:|']
    for arm,s in summaries.items():
        for q in BUDGETS:
            e=s['budgets'][str(q)]; p=e['preference']; d=e['margin_drop']; c=e['rankflip_2x2']
            lines.append(f"| {arm} | {q}% | {e['point_mean_actual_area']:.2%} / {e['cr_mean_actual_area']:.2%} | {p['point_mean']:.6f} / {p['cr_mean']:.6f} | {d['point_mean']:.6f} / {d['cr_mean']:.6f} | {e['point_rankflip_rate']:.2%} / {e['cr_rankflip_rate']:.2%} | {c['neither_flipped']} / {c['point_only_flipped']} / {c['cr_only_flipped']} / {c['both_flipped']} | {e['exact_mcnemar_pvalue']:.6g} |")
    lines += ['', 'All paired preference/margin-drop means, medians, differences, wins, Wilcoxon results and CIs are in each arm’s `summary.json`. Exact McNemar is the two-sided binomial test on discordant paired flips.','',
        '## Descriptive cross-backbone comparison','',
        '| Dataset | Backbone | n | Pointwise matched PDAUC | CR matched PDAUC | Relative improvement | Paired mean-difference95% CI |',
        '|---|---|---:|---:|---:|---:|---|']
    old_sugar=json.loads(Path('results/experiment_3/summary.json').read_text())['robustness']
    old_flickr=json.loads(Path('results/experiment_4/summary.json').read_text())['primary']
    for arm,old in [('sugarcrepe',old_sugar),('flickr30k',old_flickr)]:
        c=old.get('image_cluster_bootstrap_ci95',old['paired_bootstrap_ci95'])
        lines.append(f"| {arm} | ViT-B/32 | {old['n']} | {old['point_mean']:.6f} | {old['cr_mean']:.6f} | {old['relative_mean_improvement_percent']:.2f}% | [{c['low']:+.6f}, {c['high']:+.6f}] |")
        a=summaries[arm]['primary']
        lines.append(f"| {arm} | ViT-B/16 | 200 | {a['point_mean']:.6f} | {a['cr_mean']:.6f} | {a['relative_mean_improvement_percent']:.2f}% | {interval(a)} |")
    lines += ['', 'B/32 values are read verbatim from saved summaries; no B/32 inference or analysis was rerun. SugarCrepe B/32 uses the Experiment3 matched-area robustness result and its image-cluster CI. The table is descriptive: model-dependent eligibility, negative choices, sample size and category/image weighting differ.','',
        '## Caption and correctness overlap diagnostics','',
        f"Across all1000 common Flickr30k test images: {screens['flickr30k']['backbone_overlap']}",'',
        'Caption overlap is defined by saved caption IDs over the same ownership pool. These are descriptive diagnostics, not a new hypothesis or explanation-selection criterion.','',
        '## Technical validation and runtime','']
    runtime={}
    for arm,data in rows.items():
        runtime[arm]={'screening_seconds':screens[arm]['screening_runtime_seconds'],
            'shared_coalition_inference_seconds':sum(r['shared_inference_runtime_seconds'] for r in data),
            'point_regression_seconds':sum(r['point_regression_runtime_seconds'] for r in data),
            'cr_regression_seconds':sum(r['cr_regression_runtime_seconds'] for r in data),
            'shared_evaluation_seconds':sum(r['evaluation_runtime_seconds'] for r in data)}
        lines += [f"{arm}: a five-image technical pilot was independently verified before either full arm. All200 examples passed pinned B/16/frozen checks, screening-score/caption-vector reproduction, unchanged segmentation,128 unique shared coalitions, constrained regression/linearity, deterministic first-crossing masks and unchanged evaluator formulas. Maximum Shapley linearity error {max(r['max_abs_linearity_error'] for r in data):.3g}. Full state hashes match before/after. Independent reconstruction verified{validations[arm]['shared_coalitions_validated']:,} coalition masks and perturbed images plus embeddings, regressions, deletion rankings, budget masks and paired statistics. Protected prior files unchanged: {validations[arm]['protected_files_unchanged']:,}.",'',
            f"Runtime seconds: {runtime[arm]}.",'']
    save_json(ROOT/'runtime_summary.json',runtime)
    lines += ['Model loading, snapshot hashing, disk serialization and report generation are outside the recorded inference/evaluator costs. Both targets share coalition encoding and identical evaluator states are cached. No old-model embeddings were reused.','',
        '## Final answers and limitations','']
    for number,arm in [(1,'sugarcrepe'),(2,'flickr30k')]:
        a=summaries[arm]['primary']; conclusion_arm=('Positive mean improvement with CI above zero.' if a['mean_difference']>0 and a['paired_bootstrap_ci95']['low']>0 else
            'Positive observed mean direction, but the CI includes zero.' if a['mean_difference']>0 else 'The observed mean direction does not favor CR-SHAP.')
        lines += [f"**Question{number} — Does CR-SHAP outperform Pointwise under B/16 on {arm}?** {conclusion_arm} Relative effect {a['relative_mean_improvement_percent']:+.2f}%, Wilcoxon p={a['wilcoxon']['pvalue']:.6g}, CI {interval(a)}.",'']
    lines += [f'**Question3 — Is the direction consistent across backbones?** {conclusion}','',
        '**Question4 — Is there evidence that the advantage is specific to B/32?** '+('The positive effects in both B/16 arms argue against strict B/32 specificity in these settings; they do not establish generality across arbitrary models.' if all(positive) else 'The mixed/nonpositive B/16 results leave backbone dependence plausible, but changed task eligibility and samples prevent a clean attribution solely to architecture.'),'',
        'This fixed-scope replication uses one B/16 checkpoint, one selection seed and200 correctly ranked images per dataset. SugarCrepe image-uniform sampling differs from the older category-balanced confirmation, and Flickr30k constructs model-specific caption competitors. Approximate Kernel SHAP, SLIC granularity, mean masking, positive-only tail conventions, whole-region overshoot, and Telea perturbations constrain interpretation. There is no training, objective tuning, effect-size matching, extra baseline, grounding evaluation, dataset expansion or post-hoc robustness run.','',
        '## Artifacts','',
        'Fresh full screening/eligible pools, new caption winners and model-specific embeddings are in `data/experiment7_screening/{sugarcrepe,flickr30k}`. Selected examples, pre-generated reserves, preflight masks and technical exclusions were frozen before explanations. Each `results/experiment_7/<arm>/full` preserves configs, package/model provenance, state/source/input hashes, original images, SLIC, shared coalitions, normalized embeddings, both cosine caches, three SHAP vectors, masks, raw deletion curves, per-example metrics, statistics, timings and independent verification.','',
        'Experiment7 is complete. STOP ALL EXPERIMENTATION. No larger sample, additional seed, model, dataset, baseline, grounding experiment or Experiment8 was launched. Human review decides whether to write the paper.','']
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,(arm,data) in zip(axes,rows.items()):
        for prefix,label in [('point','Pointwise'),('cr','CR-SHAP')]:
            x=[0.]+[np.mean([r[f'{prefix}_actual_area{q}'] for r in data]) for q in BUDGETS]
            y=[np.mean([r['original_preference'] for r in data])]+[np.mean([r[f'{prefix}_preference{q}'] for r in data]) for q in BUDGETS]
            ax.plot(x,y,'o-',label=label)
        ax.set(title=arm,xlabel='Mean actual area removed',ylabel='Mean positive preference',ylim=(0,1)); ax.legend()
    fig.savefig(ROOT/'deletion_overview.png',dpi=150); plt.close(fig)
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8'); print('Final Experiment7 report generated; STOP ALL EXPERIMENTATION',flush=True)


if __name__=='__main__': main()
