import time

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr, wilcoxon

from lili_shap.core import deletion_curve, kernel_shap, positive_order


def estimate_targets(coalitions, pair_scores):
    scores = np.asarray(pair_scores, dtype=float)
    start = time.perf_counter()
    positive, pos_fit = kernel_shap(coalitions, scores[:, 0])
    positive_seconds = time.perf_counter() - start
    negative, neg_fit = kernel_shap(coalitions, scores[:, 1])
    start = time.perf_counter()
    contrastive, cr_fit = kernel_shap(coalitions, scores[:, 0] - scores[:, 1])
    cr_seconds = time.perf_counter() - start
    error = float(np.max(abs(contrastive - (positive - negative))))
    if error > 1e-10:
        raise ValueError(f'Shapley linearity failed: {error}')
    return {'pointwise': positive, 'negative': negative, 'cr_shap': contrastive}, {
        'positive': pos_fit, 'negative': neg_fit, 'contrastive': cr_fit,
        'max_abs_linearity_error': error, 'pointwise_fit_seconds': positive_seconds,
        'cr_shap_fit_seconds': cr_seconds}


def preference_deletion(image, segments, phi, pair_score, gamma):
    """Reuse validated positive-only Telea deletion; tail is integration-only."""
    cached = {}
    def score(images):
        values = np.asarray(pair_score(images), dtype=float)
        cached['pair_scores'] = values
        # Validate masks and preservation of every unmasked pixel for all states.
        order = positive_order(phi)
        mask = np.zeros(segments.shape, dtype=bool)
        for step, perturbed in enumerate(images):
            if step:
                mask |= segments == order[step-1]
            if not np.array_equal(perturbed[~mask], image[~mask]):
                raise ValueError('Deletion altered pixels outside intended SLIC mask')
        return expit(gamma * (values[:, 0] - values[:, 1]))
    curve = deletion_curve(image, segments, phi, score)
    values = cached['pair_scores']
    margins = values[:, 0] - values[:, 1]
    curve.update({'positive_cosines': values[:, 0].tolist(), 'negative_cosines': values[:, 1].tolist(),
                  'margins': margins.tolist(), 'preferences': curve['scores'],
                  'no_positive_attribution': len(curve['order']) == 0,
                  'tail_is_integration_only': True, 'num_positive_regions': len(curve['order'])})
    for percentage, threshold in [(25, .25), (50, .5)]:
        reached = np.flatnonzero(np.asarray(curve['fractions']) >= threshold)
        state = int(reached[0]) if len(reached) else None
        curve[f'state{percentage}'] = state
        curve[f'reached{percentage}'] = state is not None
        curve[f'rankflip{percentage}'] = bool(margins[state] < 0) if state is not None else None
        curve[f'margin{percentage}'] = float(margins[state]) if state is not None else None
        curve[f'margin_drop{percentage}'] = float(margins[0] - margins[state]) if state is not None else None
    return curve


def stratified_select(rows, count=100, seed=42):
    groups = {category: [] for category in sorted({r['sugarcrepe_category'] for r in rows})}
    for row in rows:
        if float(row['original_margin']) > 0:
            groups[row['sugarcrepe_category']].append(row)
    if sum(map(len, groups.values())) < count:
        raise ValueError('Not enough correctly ranked instances')
    rng = np.random.default_rng(seed)
    categories = list(groups)
    # Seeded category permutation decides who receives extra slots in water filling.
    priority = list(rng.permutation(categories))
    allocations = dict.fromkeys(categories, 0)
    while sum(allocations.values()) < count:
        eligible = [c for c in priority if allocations[c] < len(groups[c])]
        minimum = min(allocations[c] for c in eligible)
        category = next(c for c in eligible if allocations[c] == minimum)
        allocations[category] += 1
    selected_by_category = {c: [groups[c][i] for i in rng.choice(len(groups[c]), allocations[c], replace=False)]
                            for c in categories}
    # Round-robin order makes the first five pilot instances span categories.
    selected = []
    for i in range(max(allocations.values())):
        for c in priority:
            if i < len(selected_by_category[c]):
                selected.append(selected_by_category[c][i])
    return selected, allocations


def group_statistics(rows):
    point = np.asarray([r['pointwise_pdauc'] for r in rows], dtype=float)
    cr = np.asarray([r['cr_shap_pdauc'] for r in rows], dtype=float)
    difference = point - cr
    result = {'n': len(rows), 'pointwise_mean_pdauc': float(point.mean()),
        'pointwise_median_pdauc': float(np.median(point)), 'cr_shap_mean_pdauc': float(cr.mean()),
        'cr_shap_median_pdauc': float(np.median(cr)), 'mean_paired_difference': float(difference.mean()),
        'median_paired_difference': float(np.median(difference)),
        'relative_mean_improvement_percent': float(100 * difference.mean() / point.mean()) if point.mean() else None,
        'cr_shap_wins': int((difference > 0).sum()), 'pointwise_wins': int((difference < 0).sum()),
        'ties': int((difference == 0).sum()), 'rankflip': {}, 'margin25': {}}
    for threshold in [25, 50]:
        entry = {}
        for method in ['pointwise', 'cr_shap']:
            eligible = [r for r in rows if r[f'{method}_rankflip{threshold}'] is not None]
            flips = sum(r[f'{method}_rankflip{threshold}'] for r in eligible)
            entry[method] = {'n_reached': len(eligible), 'n_unreached': len(rows)-len(eligible),
                'n_flipped': int(flips), 'percent_among_reached': 100*flips/len(eligible) if eligible else None,
                'percent_of_all_with_observed_flip': 100*flips/len(rows)}
        common = [r for r in rows if r[f'pointwise_rankflip{threshold}'] is not None and r[f'cr_shap_rankflip{threshold}'] is not None]
        entry['paired_common_coverage'] = {'n': len(common), **{m: 100*sum(r[f'{m}_rankflip{threshold}'] for r in common)/len(common)
            if common else None for m in ['pointwise', 'cr_shap']}}
        result['rankflip'][str(threshold)] = entry
    for method in ['pointwise', 'cr_shap']:
        eligible = [r for r in rows if r[f'{method}_margin25'] is not None]
        result['margin25'][method] = {'n_reached': len(eligible),
            'mean_margin': float(np.mean([r[f'{method}_margin25'] for r in eligible])) if eligible else None,
            'mean_margin_drop': float(np.mean([r[f'{method}_margin_drop25'] for r in eligible])) if eligible else None}
    result['no_positive_attribution'] = {m: sum(r[f'{m}_num_positive_regions'] == 0 for r in rows)
                                         for m in ['pointwise', 'cr_shap']}
    return result


def analyze(rows, seed=42):
    overall = group_statistics(rows)
    differences = np.asarray([r['pdauc_difference'] for r in rows])
    if np.all(differences == 0):
        statistic, pvalue = 0., 1.
    else:
        test = wilcoxon(differences, alternative='two-sided', zero_method='wilcox', method='auto')
        statistic, pvalue = float(test.statistic), float(test.pvalue)
    overall['wilcoxon'] = {'statistic': statistic, 'pvalue': pvalue, 'alternative': 'two-sided',
                           'multiple_testing_correction': None}
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(rows), size=(10000, len(rows)))].mean(axis=1)
    overall['paired_bootstrap_mean_difference_ci95'] = {'low': float(np.percentile(boot, 2.5)),
        'high': float(np.percentile(boot, 97.5)), 'resamples': 10000, 'seed': seed, 'method': 'percentile'}
    categories = {c: group_statistics([r for r in rows if r['sugarcrepe_category'] == c])
                  for c in sorted({r['sugarcrepe_category'] for r in rows})}
    margin = np.asarray([r['original_margin'] for r in rows])
    correlation = spearmanr(margin, differences) if np.ptp(margin) > 0 and np.ptp(differences) > 0 else None
    return {'overall': overall, 'categories': categories,
        'original_margin_relationship': {'exploratory': True,
            'spearman_rho': float(correlation.statistic) if correlation else None,
            'pvalue': float(correlation.pvalue) if correlation else None},
        'threshold_convention': 'Unreached actual deletion thresholds are missing; report reached-case and common-case denominators.'}


def figure_selection(rows):
    improved = sorted([r for r in rows if r['pdauc_difference'] > 0], key=lambda r: (-r['pdauc_difference'], r['example_id']))[:5]
    worsened = sorted([r for r in rows if r['pdauc_difference'] < 0], key=lambda r: (r['pdauc_difference'], r['example_id']))[:5]
    used = {r['example_id'] for r in improved + worsened}
    equal = sorted([r for r in rows if r['example_id'] not in used], key=lambda r: (abs(r['pdauc_difference']), r['example_id']))[:5]
    return {'largest_improvements': [r['example_id'] for r in improved],
        'closest_to_equal': [r['example_id'] for r in equal],
        'largest_worsenings': [r['example_id'] for r in worsened]}
