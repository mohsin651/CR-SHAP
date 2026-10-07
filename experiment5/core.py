import numpy as np
from scipy.stats import binomtest

from experiment4.core import paired_statistics
from experiment3.core import BUDGETS

METHODS = ('shap_point', 'shap_cr', 'grad_point', 'grad_cr')
COMPARISONS = {
    'A_SHAP_objective': ('shap_point', 'shap_cr'),
    'B_GradECLIP_objective': ('grad_point', 'grad_cr'),
    'C_pointwise_families': ('shap_point', 'grad_point'),
    'D_contrastive_families': ('shap_cr', 'grad_cr')}


def paired(rows, first, second, suffix, higher_better=False):
    result = paired_statistics(rows, first+'_'+suffix, second+'_'+suffix, higher_better)
    result['first_method'], result['second_method'] = first, second
    result['first_mean'], result['second_mean'] = result.pop('point_mean'), result.pop('cr_mean')
    result['first_median'], result['second_median'] = result.pop('point_median'), result.pop('cr_median')
    result['second_wins'], result['first_wins'] = result.pop('cr_wins'), result.pop('point_wins')
    result['difference_sign'] = 'Second minus first' if higher_better else 'First minus second'
    return result


def analyze(rows):
    if len(rows) != 300 or len({r['image_id'] for r in rows}) != 300:
        raise ValueError('Exactly 300 frozen unique query images required')
    summary = {'primary_endpoint': 'MatchedArea-PDAUC-50', 'comparisons': {}, 'methods': {},
        'testing': 'B is key new comparison; A reused. C/D and budgets secondary; two-sided unadjusted p-values; paired percentile 10000 draws seed42.'}
    for method in METHODS:
        values = [r[method+'_matched_pdauc50'] for r in rows]
        summary['methods'][method] = {'matched_mean': float(np.mean(values)), 'matched_median': float(np.median(values)),
            'positive_mean': float(np.mean([r[method+'_positive_pdauc'] for r in rows]))}
    for name, (first, second) in COMPARISONS.items():
        comparison = {'primary': paired(rows, first, second, 'matched_pdauc50'),
            'positive_only': paired(rows, first, second, 'positive_pdauc'), 'budgets': {}}
        for q in BUDGETS:
            a = np.asarray([r[f'{first}_rankflip{q}'] for r in rows], dtype=bool)
            b = np.asarray([r[f'{second}_rankflip{q}'] for r in rows], dtype=bool)
            first_only, second_only = int((a & ~b).sum()), int((~a & b).sum())
            comparison['budgets'][str(q)] = {'preference': paired(rows, first, second, f'preference{q}'),
                'margin_drop': paired(rows, first, second, f'margin_drop{q}', True),
                'first_rankflip_rate': float(a.mean()), 'second_rankflip_rate': float(b.mean()),
                'outcome_counts': {'neither': int((~a & ~b).sum()), 'first_only': first_only,
                                  'second_only': second_only, 'both': int((a & b).sum())},
                'exact_mcnemar_pvalue': float(binomtest(first_only,first_only+second_only,.5).pvalue) if first_only+second_only else 1.}
        summary['comparisons'][name] = comparison
    return summary
