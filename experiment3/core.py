from collections import Counter

import cv2
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest, wilcoxon

from lili_shap.core import mean_mask

BUDGETS = (10, 20, 30, 40, 50)


def select_sample(rows, excluded, seed=2026, target=100):
    rng = np.random.default_rng(seed)
    selected, categories = [], {}
    for category in sorted({r['sugarcrepe_category'] for r in rows}):
        eligible = [r for r in rows if r['sugarcrepe_category'] == category and float(r['original_margin']) > 0]
        pool = [r for r in eligible if r['example_id'] not in excluded]
        groups = {}
        for row in sorted(pool, key=lambda r: r['example_id']):
            groups.setdefault(row['image_id'], []).append(row)
        images = sorted(groups)
        n = min(target, len(pool))
        chosen = []
        for i in rng.permutation(len(images))[:n]:
            candidates = groups[images[i]]
            chosen.append(candidates[int(rng.integers(len(candidates)))])
        if len(chosen) < n:
            used = {r['example_id'] for r in chosen}
            remaining = [r for r in pool if r['example_id'] not in used]
            chosen.extend(remaining[i] for i in rng.choice(len(remaining), n-len(chosen), replace=False))
        selected.extend(chosen)
        categories[category] = {'eligible_before_exclusion': len(eligible), 'excluded_experiment2': len(eligible)-len(pool),
            'available_new_instances': len(pool), 'available_new_images': len(images), 'selected': n,
            'shortage': target-n, 'selected_unique_images': len({r['image_id'] for r in chosen})}
    # Fixed round-robin order makes the initial validation span categories.
    groups = {c: [r for r in selected if r['sugarcrepe_category'] == c] for c in categories}
    selected = [groups[c][i] for i in range(max(map(len, groups.values()))) for c in groups if i < len(groups[c])]
    counts = Counter(r['image_id'] for r in selected)
    return selected, {'seed': seed, 'target_per_category': target, 'categories': categories,
        'excluded_experiment2_instances': sum(c['excluded_experiment2'] for c in categories.values()),
        'n': len(selected), 'unique_images': len(counts), 'images_repeated': sum(n>1 for n in counts.values()),
        'maximum_instances_per_image': max(counts.values()),
        'rule': 'Uniform unique-image sampling within each category, then uniform instance within image; fill remaining slots from unused instances only if fewer than target images.'}


def matched_deletion(image, segments, phi, pair_score, gamma):
    order = sorted(range(len(phi)), key=lambda j: (-phi[j], j))
    mask = np.zeros(segments.shape, np.uint8)
    images, masks, fractions = [image], [mask.copy()], [0.]
    for j in order:
        mask[segments == j] = 255
        perturbed = mean_mask(image, mask) if np.all(mask) else cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        if not np.array_equal(perturbed[mask == 0], image[mask == 0]):
            raise ValueError('Telea changed an unmasked pixel')
        images.append(perturbed)
        masks.append(mask.copy())
        fractions.append(float(np.mean(mask > 0)))
    scores = np.asarray(pair_score(images), dtype=float)
    margins = scores[:, 0]-scores[:, 1]
    preferences = expit(gamma*margins)
    states = [int(np.flatnonzero(np.asarray(fractions) >= q/100)[0]) for q in BUDGETS]
    x = np.asarray([0.] + [fractions[s] for s in states])
    y = preferences[[0]+states]
    curve = {'order': order, 'fractions': fractions, 'positive_cosines': scores[:,0].tolist(),
        'negative_cosines': scores[:,1].tolist(), 'margins': margins.tolist(), 'preferences': preferences.tolist(),
        'budget_states': dict(zip(map(str, BUDGETS), states)), 'auc_fractions': x.tolist(),
        'auc_preferences': y.tolist(), 'matched_pdauc50': float(np.trapz(y,x)/x[-1]),
        'no_extrapolation': True}
    return curve, np.asarray(masks, dtype=np.uint8)


def paired_statistics(rows, point_key, cr_key, higher_better=False):
    point = np.asarray([r[point_key] for r in rows], dtype=float)
    cr = np.asarray([r[cr_key] for r in rows], dtype=float)
    diff = cr-point if higher_better else point-cr
    test = wilcoxon(diff, alternative='two-sided', zero_method='wilcox', method='auto') if np.any(diff) else None
    rng = np.random.default_rng(42)
    bootstrap = diff[rng.integers(0,len(rows),size=(10000,len(rows)))].mean(axis=1)
    images = sorted({r['image_id'] for r in rows})
    groups = {image: i for i,image in enumerate(images)}
    indices = np.asarray([groups[r['image_id']] for r in rows])
    sums = np.bincount(indices,weights=diff,minlength=len(images))
    counts = np.bincount(indices,minlength=len(images))
    rng = np.random.default_rng(42)
    draws = rng.integers(0,len(images),size=(10000,len(images)))
    cluster = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    def ci(values):
        return {'low':float(np.percentile(values,2.5)), 'high':float(np.percentile(values,97.5)),
                'resamples':10000, 'seed':42, 'method':'percentile'}
    return {'n':len(rows), 'point_mean':float(point.mean()), 'point_median':float(np.median(point)),
        'cr_mean':float(cr.mean()), 'cr_median':float(np.median(cr)), 'mean_difference':float(diff.mean()),
        'median_difference':float(np.median(diff)), 'relative_mean_improvement_percent':float(100*diff.mean()/point.mean()) if point.mean() else None,
        'cr_wins':int((diff>0).sum()), 'point_wins':int((diff<0).sum()), 'ties':int((diff==0).sum()),
        'wilcoxon':{'statistic':float(test.statistic) if test else 0., 'pvalue':float(test.pvalue) if test else 1., 'alternative':'two-sided'},
        'paired_bootstrap_ci95':ci(bootstrap), 'image_cluster_bootstrap_ci95':ci(cluster),
        'difference_sign':'CR minus Pointwise' if higher_better else 'Pointwise minus CR'}


def analyze(rows):
    result = {'primary':paired_statistics(rows,'point_positive_pdauc','cr_positive_pdauc'),
        'robustness':paired_statistics(rows,'point_matched_pdauc50','cr_matched_pdauc50'), 'budgets':{}, 'categories':{},
        'secondary_testing':'Two-sided unadjusted exploratory robustness tests; primary endpoint pre-declared; cluster CIs preferred.'}
    for q in BUDGETS:
        point = np.asarray([r[f'point_rankflip{q}'] for r in rows],dtype=bool)
        cr = np.asarray([r[f'cr_rankflip{q}'] for r in rows],dtype=bool)
        b,c = int((point & ~cr).sum()),int((~point & cr).sum())
        result['budgets'][str(q)] = {'preference':paired_statistics(rows,f'point_preference{q}',f'cr_preference{q}'),
            'margin_drop':paired_statistics(rows,f'point_margin_drop{q}',f'cr_margin_drop{q}',True),
            'point_mean_actual_area':float(np.mean([r[f'point_actual_area{q}'] for r in rows])),
            'cr_mean_actual_area':float(np.mean([r[f'cr_actual_area{q}'] for r in rows])),
            'point_rankflip_rate':float(point.mean()),'cr_rankflip_rate':float(cr.mean()),
            'rankflip_2x2':{'both_not_flipped':int((~point & ~cr).sum()),'point_only_flipped':b,'cr_only_flipped':c,'both_flipped':int((point & cr).sum())},
            'exact_mcnemar_pvalue':float(binomtest(b,b+c,.5,alternative='two-sided').pvalue) if b+c else 1.}
    for category in sorted({r['category'] for r in rows}):
        subset = [r for r in rows if r['category']==category]
        result['categories'][category] = {endpoint:paired_statistics(subset,point,cr) for endpoint,point,cr in
            [('primary','point_positive_pdauc','cr_positive_pdauc'),('robustness','point_matched_pdauc50','cr_matched_pdauc50')]}
    return result
