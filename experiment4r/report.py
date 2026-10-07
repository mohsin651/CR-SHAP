"""Census/remainder robustness report; original Experiment4 is preserved."""
import json
from pathlib import Path

import numpy as np

from experiment2.io import save_json,sha256,verify_snapshot,write_csv
from .core import BUDGETS,analyze,partition
from .prepare import CONFIG,INPUT,ORIGINAL,ROOT


def interval(s):
    c=s['paired_bootstrap_ci95']; return f"[{c['low']:+.6f}, {c['high']:+.6f}]"


def bootstrap_samples(rows):
    # Preserve actual paired resample means, not only the resulting interval.
    difference=np.asarray([r['matched_pdauc_difference'] for r in rows])
    rng=np.random.default_rng(42)
    return difference[rng.integers(0,len(rows),size=(10000,len(rows)))].mean(axis=1)


def main():
    if (ROOT/'report.md').exists(): raise FileExistsError(ROOT/'report.md')
    newroot=ROOT/'new_full'; validation=json.loads((newroot/'verification.json').read_text())
    if not validation['all_passed']: raise ValueError('New remainder must be verified')
    registry=json.loads((ROOT/'reused_artifacts.json').read_text())
    for entry in registry:
        for name,digest in entry['files'].items():
            if sha256(Path(entry['artifact_directory'])/name)!=digest: raise ValueError('Reused original artifact changed')
    oldrows=json.loads((ORIGINAL/'rows.json').read_text()); newrows=json.loads((newroot/'rows.json').read_text())
    oldsummary=json.loads((ORIGINAL/'summary.json').read_text())
    if analyze(oldrows)!=oldsummary: raise ValueError('Original300 no longer reproduce exactly')
    fullrows=oldrows+newrows
    intended=json.loads((ROOT/'population_manifest.json').read_text())
    screening=json.loads(Path('data/flickr30k_screening_exp4/rows.json').read_text())
    population,remainder,digest=partition(screening,[r['image_id'] for r in oldrows])
    if population!=intended or digest!=json.loads((ROOT/'scope.json').read_text())['eligible_id_set_sha256']:
        raise ValueError('Original793 population changed')
    if len({r['image_id'] for r in fullrows})!=len(fullrows): raise ValueError('Duplicate population queries')
    preflight=json.loads((ROOT/'technical_failures_preflight.json').read_text()); execution=json.loads((newroot/'technical_failures.json').read_text())
    failures=preflight+execution
    if len(fullrows)+len(failures)!=793 or len(newrows)+len(failures)!=493: raise ValueError('Unaccounted eligible query')
    if set(r['image_id'] for r in fullrows)|set(r['image_id'] for r in failures)!=set(r['image_id'] for r in intended):
        raise ValueError('Intended/evaluable/failure set mismatch')
    full=analyze(fullrows); unseen=analyze(newrows)
    save_json(ROOT/'rows.json',fullrows); write_csv(ROOT/'results.csv',fullrows)
    save_json(ROOT/'summary.json',{'full_eligible_evaluable':full,'previously_unevaluated_remainder':unseen,'original_sample_reused':oldsummary})
    save_json(ROOT/'technical_failures.json',failures)
    artifact_sources=[{'image_id':r['image_id'],'provenance':'reused-original300','artifact_directory':entry['artifact_directory']} for r,entry in zip(oldrows,registry)]
    selected=json.loads((INPUT/'selected_images.json').read_text()); index={r['image_id']:i for i,r in enumerate(selected)}
    artifact_sources.extend({'image_id':r['image_id'],'provenance':'newly-computed','artifact_directory':str(newroot/f"example_{index[r['image_id']]:03d}")} for r in newrows)
    save_json(ROOT/'artifact_sources.json',artifact_sources)
    np.savez_compressed(ROOT/'primary_bootstrap_resample_means.npz',full=bootstrap_samples(fullrows),remainder=bootstrap_samples(newrows))
    old=oldsummary['primary']; total=full['primary']; rest=unseen['primary']
    change=total['mean_difference']-old['mean_difference']; relative_change=total['relative_mean_improvement_percent']-old['relative_mean_improvement_percent']
    save_json(ROOT/'sample_vs_population.json',{'absolute_effect_full_minus_original':change,'relative_effect_percentage_points_full_minus_original':relative_change,
        'analysis':'Descriptive nested-population comparison, no independent-sample significance test'})
    positive=total['mean_difference']>0
    opening=('CR-SHAP has lower observed mean matched-area PDAUC across all793 originally eligible queries.' if positive and len(fullrows)==793 else
        'CR-SHAP has lower observed mean matched-area PDAUC among evaluable eligible queries; documented failures prevent a complete793-query census.' if positive else
        'The observed full-evaluable aggregate does not favor CR-SHAP; the frozen protocol was not adjusted.')
    lines=['# Experiment4R: full eligible Flickr30k robustness analysis','',opening,'',
        'This is a post-specified full eligible-test-population robustness analysis. The original pre-specified/random-sample Experiment4 result remains unchanged and is reported separately. It is not replaced by this extension.','',
        '## Exact population and integrity','',
        f"Original screening:1000 Karpathy test images and5000 owned captions,793 strictly correctly ranked queries,207 incorrect,0 exact ties. Original300 and remaining493 are disjoint and their union is exactly793. Canonical eligible-ID checksum: `{digest}` (SHA256 of compact UTF8 JSON of lexicographically sorted IDs). No new retrieval screening, model-defined captions, eligibility, sampling seed or reserve list was introduced.",'',
        f"Intended census=793; original reused=300; new intended=493; new evaluable={len(newrows)}; total evaluable={len(fullrows)}; technical failures={len(failures)}. Every failure and its exact phase/reason is retained in `technical_failures.json`. No failed query was replaced, and no outcome-based exclusion was used.",'',
        'All original300 artifact files were hashed before reuse and checked again afterwards. Five original examples were independently reconstructed using the existing verifier with its output redirected into this new directory. The original300 per-example metrics and complete statistical summary reproduce exactly in their original order. Their images, SLIC, coalitions, score caches, SHAP vectors and deletion curves are referenced in place, never recomputed or rewritten.','',
        '## Unchanged Experiment4 protocol','',
        'Model `openai/clip-vit-base-patch32`, commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, frozen eval mode. T+ and T−, caption IDs/owners, and scores come only from the original B/32 screening. B/16 competitors and embeddings are not used. Gamma remains100 from the same model logit scale. All method settings are preserved verbatim in `original_protocol_config.json`; its sampling description refers to the original300 study, while `scope.json` defines this census extension.','',
        'EXIF transpose, RGB, aspect-preserving max-side512, unchanged CLIP processor/tokenizer. SLIC:16 requested regions, compactness30, start_label0, channel_axis−1, scikit-image0.24.0 defaults. Mean-RGB masking;128 unique shared coalitions seed42 with mandatory endpoints/singletons/complements; unchanged constrained Kernel SHAP. One normalized coalition image embedding provides both caption cosine scores, with negative-caption regression validating Shapley linearity. All new feasibility checks finished before new explanation outcomes.','',
        'Primary MatchedArea-PDAUC-50: all signed region values descending, ascending region-index ties; first whole-region crossings at10/20/30/40/50%; actual removed area; Telea radius3 on the original image and unchanged all-mask RGB-mean fallback. Preference=sigmoid(gamma × cosine margin). Integrate original and five crossing states, divide by final actual fraction, retain repeated crossings with zero-width intervals, no extrapolated tail. Secondary positive-only PDAUC preserves the original strictly-positive ranking and constant terminal integration tail to100%.','',
        '## Critical three-way primary comparison','',
        '| Population | n | Pointwise mean / median | CR mean / median | Absolute mean / median difference | Relative improvement | CR / Point wins / ties | CR win rate | Paired95% CI | Wilcoxon statistic / p |',
        '|---|---:|---|---|---|---:|---|---:|---|---|']
    tables=[]
    for label,a in [('Original sampled Experiment4',old),('Previously unevaluated remainder',rest),('Complete eligible population' if len(fullrows)==793 else 'Eligible population, evaluable subset',total)]:
        lines.append(f"| {label} | {a['n']} | {a['point_mean']:.6f} / {a['point_median']:.6f} | {a['cr_mean']:.6f} / {a['cr_median']:.6f} | {a['mean_difference']:+.6f} / {a['median_difference']:+.6f} | {a['relative_mean_improvement_percent']:.2f}% | {a['cr_wins']} / {a['point_wins']} / {a['ties']} | {a['cr_wins']/a['n']:.2%} | {interval(a)} | {a['wilcoxon']['statistic']:.6g} / {a['wilcoxon']['pvalue']:.6g} |")
        tables.append({'population':label,**{k:v for k,v in a.items() if not isinstance(v,dict)},'ci_low':a['paired_bootstrap_ci95']['low'],'ci_high':a['paired_bootstrap_ci95']['high'],'wilcoxon_p':a['wilcoxon']['pvalue']})
    write_csv(ROOT/'population_comparison.csv',tables)
    lines += ['',f"Full minus original absolute paired effect: {change:+.6f}; relative improvement change: {relative_change:+.2f} percentage points. The original300 are contained in the census; no independent-samples comparison or inappropriate p-value between them was calculated.",'',
        'The remainder is previously unevaluated under the Experiment4 B/32 protocol, not a newly randomized confirmation sample. No settings were changed after viewing its result.','',
        '## Finite population versus broader inference','',
        'For a complete793-query census, the observed finite-population mean and effect are directly known; a bootstrap CI is not needed to estimate that already-observed mean. For comparability, paired two-sided Wilcoxon and10000 paired percentile bootstrap resamples, seed42, describe paired variability and potential broader generalization uncertainty under exchangeability assumptions. They are not a probability-sampling guarantee about other datasets, models, or all retrieval queries. Eligibility conditions on correct retrieval, and candidate-caption sharing also limits simple generalization. If failures exist, descriptive results apply only to the stated evaluable population.','',
        '## Secondary positive-only PDAUC','',
        '| Population | Point mean / median | CR mean / median | Mean difference | Relative improvement | CR / Point wins / ties | 95% CI | Wilcoxon p |','|---|---|---|---:|---:|---|---|---:|']
    for label,summary in [('Original300',oldsummary),('Remainder',unseen),('Full eligible evaluable',full)]:
        a=summary['secondary_positive_only']
        lines.append(f"| {label} | {a['point_mean']:.6f} / {a['point_median']:.6f} | {a['cr_mean']:.6f} / {a['cr_median']:.6f} | {a['mean_difference']:+.6f} | {a['relative_mean_improvement_percent']:.2f}% | {a['cr_wins']} / {a['point_wins']} / {a['ties']} | {interval(a)} | {a['wilcoxon']['pvalue']:.6g} |")
    lines += ['', '## Fixed-budget and RankFlip robustness','',
        '| Population | Budget | Actual area Point / CR | Preference Point / CR | Margin drop Point / CR | RankFlip Point / CR | Neither / Point only / CR only / Both | Exact McNemar p |',
        '|---|---:|---|---|---|---|---|---:|']
    budget_rows=[]
    for label,summary in [('Full eligible evaluable',full),('Previously unevaluated remainder',unseen)]:
        for q in BUDGETS:
            e=summary['budgets'][str(q)]; p=e['preference']; d=e['margin_drop']; c=e['rankflip_2x2']
            lines.append(f"| {label} | {q}% | {e['point_mean_actual_area']:.2%} / {e['cr_mean_actual_area']:.2%} | {p['point_mean']:.6f} / {p['cr_mean']:.6f} | {d['point_mean']:.6f} / {d['cr_mean']:.6f} | {e['point_rankflip_rate']:.2%} / {e['cr_rankflip_rate']:.2%} | {c['neither_flipped']} / {c['point_only_flipped']} / {c['cr_only_flipped']} / {c['both_flipped']} | {e['exact_mcnemar_pvalue']:.6g} |")
            budget_rows.append({'population':label,'budget':q,**{k:v for k,v in e.items() if not isinstance(v,dict)},**c})
    write_csv(ROOT/'budget_results.csv',budget_rows)
    lines += ['', 'All budgets are reported without cherry-picking. Exact two-sided McNemar is the binomial test on discordant paired flips. Full paired preference/margin-drop/positive-only analyses remain in `summary.json`; these secondary p-values are unadjusted. Positive AUC/preference differences mean Pointwise minus CR; positive margin-drop and flip-rate differences mean CR minus Pointwise.','',
        '## Required answers','',
        f"**1. Does CR have lower matched-area PDAUC across the original eligible population?** {'Yes, descriptively' if positive else 'No, the observed aggregate does not favor CR'}: Pointwise {total['point_mean']:.6f}, CR {total['cr_mean']:.6f}, paired mean difference {total['mean_difference']:+.6f}, n={total['n']}. Intended n=793; any missing coverage is explicitly reported above.",'',
        f"**2. Full effect versus the original+9.19%?** The full-evaluable effect is {total['relative_mean_improvement_percent']:+.2f}%, a change of {relative_change:+.2f} percentage points; absolute effect changed by {change:+.6f}.",'',
        f"**3. Does the advantage persist on the previously unevaluated remainder?** {'The observed mean still favors CR' if rest['mean_difference']>0 else 'The observed mean does not favor CR'}: {rest['relative_mean_improvement_percent']:+.2f}%, n={rest['n']}, CI {interval(rest)}, Wilcoxon p={rest['wilcoxon']['pvalue']:.6g}. This is a remainder diagnostic, not an independently randomized confirmation.",'',
        '**4. Does RankFlip retain the qualitative advantage?** '+('CR has a higher observed flip rate at every specified budget in the full-evaluable population.' if all(full['budgets'][str(q)]['cr_rankflip_rate']>full['budgets'][str(q)]['point_rankflip_rate'] for q in BUDGETS) else 'The full-population budget table shows mixed or tied flip-rate directions; no budget is omitted.')+' Exact McNemar results are provided separately for all budgets.','',
        '**5. Was the original sample reasonably representative?** '+('The positive direction persists across the full-evaluable population and remainder. The magnitude comparison above describes how much the original estimate differs; this is not a formal equivalence test or proof of representativeness.' if positive and rest['mean_difference']>0 else 'The full/remainder findings weaken the original sample’s representativeness; the original estimate should not be generalized without this qualification.'),'',
        '## Validation, cost and artifacts','',
        f"Both newly computed pilot and full remainder were independently checked. Verified new queries={validation['n']}; coalition masks/perturbations={validation['shared_coalitions_validated']:,}; maximum linearity error={validation['max_abs_linearity_error']:.3g}. Full model state hashes match before/after. Original300 reused metrics and summary match exactly; all prior source/results are protected by before/after hashes.",'']
    runtime={'new_shared_coalition_inference_seconds':sum(r['shared_inference_runtime_seconds'] for r in newrows),
        'new_point_regression_seconds':sum(r['point_regression_runtime_seconds'] for r in newrows),
        'new_cr_regression_seconds':sum(r['cr_regression_runtime_seconds'] for r in newrows),
        'new_evaluation_seconds':sum(r['evaluation_runtime_seconds'] for r in newrows),'reused_original_queries':300,'new_evaluated_queries':len(newrows)}
    save_json(ROOT/'runtime_summary.json',runtime)
    lines += [f'New-computation runtime seconds: {runtime}. Original300 inference cost was not incurred again. Model loading, integrity reconstruction, hashing and report generation are outside these per-example timings.','',
        'Manifests, population hash, exact original config, pre-run checks and original artifact hashes are at the root. `artifact_sources.json` distinguishes reused paths from new paths. New artifacts are in `new_full/`; original300 remain in their original directories. Curves, original images, SLIC, coalitions, embeddings, scores, all phi vectors, positive/matched masks, fit checks, per-example metrics, environment, state hashes, failure evidence, subgroup/full summaries and bootstrap resample means are preserved.','',
        'This remains conditional evidence under one frozen B/32 Flickr30k/Telea protocol. Bounding-image ownership does not ensure semantic falsity of a competing caption, whole-region budget overshoot remains, and128-coalition SHAP is approximate. It does not establish universal superiority, causal faithfulness, or generalization to arbitrary settings. Earlier B/16/SugarCrepe results are separate evidence and were not recomputed.','',
        'Experiment4R is complete. STOP. No incorrectly ranked query, new seed, larger B/16/SugarCrepe study, new dataset/model/method/grounding analysis or Experiment8 was evaluated. Wait for human review before paper writing.','']
    protected=json.loads((ROOT/'protected_experiments.json').read_text()); verify_snapshot(protected)
    save_json(ROOT/'verification.json',{'all_passed':True,'intended':793,'evaluable':len(fullrows),'reused_exact':300,
        'new_evaluable':len(newrows),'technical_failures':len(failures),'eligible_id_set_sha256':digest,'protected_files_unchanged':len(protected),
        'original_summary_exact':True,'no_new_retrieval':True,'no_original300_inference':True})
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8'); print('Experiment4R census report complete; STOP',flush=True)


if __name__=='__main__': main()
