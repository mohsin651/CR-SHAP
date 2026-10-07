import math

import cv2
import numpy as np
from scipy.stats import wilcoxon


METHODS = ('mean', 'lama', 'lama_expanded')


def sample_coalitions(d, count=128, seed=42):
    """Unique coalitions; mandatory endpoints, singletons, and complements.

    Remaining coalitions are uniform over the binary cube. Kernel weights
    therefore use the ordinary Shapley kernel, without proposal correction.
    """
    if d < 2 or count < 2 + 2 * d or count > 2 ** d:
        raise ValueError(f'Cannot sample {count} unique coalitions for {d} features')
    rows = [tuple([0] * d), tuple([1] * d)]
    for j in range(d):
        row = np.zeros(d, dtype=np.uint8)
        row[j] = 1
        rows.extend([tuple(row), tuple(1 - row)])
    rows = list(dict.fromkeys(rows))
    seen = set(rows)
    rng = np.random.default_rng(seed)
    while len(rows) < count:
        row = tuple(rng.integers(0, 2, size=d))
        if row not in seen:
            seen.add(row)
            rows.append(row)
    return np.asarray(rows, dtype=np.uint8)


def kernel_shap(coalitions, scores):
    """Weighted least squares with exact empty/full endpoint constraints."""
    z = np.asarray(coalitions, dtype=float)
    y = np.asarray(scores, dtype=float)
    d = z.shape[1]
    empty = np.flatnonzero(z.sum(1) == 0)
    full = np.flatnonzero(z.sum(1) == d)
    if len(empty) != 1 or len(full) != 1 or not np.isfinite(y).all():
        raise ValueError('Require one empty/full coalition and finite scores')
    base = float(y[empty[0]])
    total = float(y[full[0]] - base)
    interior = (z.sum(1) > 0) & (z.sum(1) < d)
    x = z[interior]
    sizes = x.sum(1).astype(int)
    w = np.array([(d - 1) / (math.comb(d, int(k)) * k * (d-k)) for k in sizes])
    # Eliminate the final coefficient: phi[-1] = total - sum(phi[:-1]).
    design = x[:, :-1] - x[:, -1:]
    target = y[interior] - base - x[:, -1] * total
    root = np.sqrt(w / w.max())
    coef, _, rank, _ = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
    if rank != d - 1:
        raise ValueError('Coalitions do not identify all SHAP coefficients')
    phi = np.r_[coef, total - coef.sum()]
    residual = y[interior] - (base + x @ phi)
    return phi, {'base_value': base, 'full_value': base + total,
                 'efficiency_residual': float(phi.sum() - total),
                 'weighted_rmse': float(np.sqrt(np.average(residual ** 2, weights=w)))}


def removal_mask(segments, coalition, radius=0):
    mask = (np.asarray(coalition)[segments] == 0).astype(np.uint8) * 255
    if radius:
        mask = cv2.dilate(mask, np.ones((2*radius+1, 2*radius+1), np.uint8))
    return mask


def mean_mask(image, mask):
    result = image.copy()
    result[mask > 0] = np.rint(image.mean(axis=(0, 1))).astype(np.uint8)
    return result


def positive_order(phi):
    phi = np.asarray(phi)
    return np.asarray(sorted(np.flatnonzero(phi > 0), key=lambda j: (-phi[j], j)), dtype=int)


def deletion_curve(image, segments, phi, score):
    """AUC on [0,1] of actual deleted pixel area, using only positive SHAP.

    After all positive features are removed, extend the last score constantly
    to x=1. This gives every explanation the same integration domain without
    silently deleting zero/negative features. Save the unextended curve too.
    """
    order = positive_order(phi)
    images = [image]
    fractions = [0.0]
    mask = np.zeros(segments.shape, dtype=np.uint8)
    for j in order:
        mask[segments == j] = 255
        # Telea cannot infer an image with no remaining context. Use the same
        # explicit global-mean fallback for that endpoint for every method.
        perturbed = mean_mask(image, mask) if np.all(mask) else cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        images.append(perturbed)
        fractions.append(float(np.mean(mask > 0)))
    values = np.asarray(score(images), dtype=float)
    x = np.asarray(fractions)
    auc_x, auc_y = x, values
    if x[-1] < 1:
        auc_x = np.r_[x, 1.0]
        auc_y = np.r_[values, values[-1]]
    return {'order': order.tolist(), 'fractions': x.tolist(), 'scores': values.tolist(),
            'auc_fractions': auc_x.tolist(), 'auc_scores': auc_y.tolist(),
            'positive_area_fraction': float(x[-1]),
            'auc': float(np.trapz(auc_y, auc_x))}


def summarize(rows):
    arrays = {m: np.array([r[m + '_shap_deletion_auc'] for r in rows]) for m in METHODS}
    result = {'n_images': len(rows), 'methods': {}, 'paired_differences': {}, 'wilcoxon': {}}
    for m, a in arrays.items():
        result['methods'][m] = {'mean_auc': float(a.mean()), 'median_auc': float(np.median(a))}
    for first, second in [('mean', 'lama'), ('mean', 'lama_expanded'), ('lama', 'lama_expanded')]:
        diff = arrays[first] - arrays[second]
        key = first + '_minus_' + second
        result['paired_differences'][key] = {'values': diff.tolist(), 'mean': float(diff.mean()),
            'median': float(np.median(diff)), 'wins_for_second': int((diff > 0).sum()),
            'ties': int((diff == 0).sum())}
        if first == 'mean':
            if np.all(diff == 0):
                stat, p = 0.0, 1.0
            else:
                test = wilcoxon(diff, alternative='two-sided', zero_method='wilcox', method='auto')
                stat, p = float(test.statistic), float(test.pvalue)
            result['wilcoxon'][key] = {'statistic': stat, 'pvalue': p, 'alternative': 'two-sided',
                                      'zero_method': 'wilcox', 'multiple_testing_correction': None}
    result['interpretation'] = 'Exploratory; qualitative review required. Positive paired differences favor the second method.'
    return result
