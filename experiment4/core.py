"""Retrieval ownership, frozen sampling, and paired inference for Experiment 4."""
from collections import Counter

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

from experiment3.core import BUDGETS, matched_deletion


def validate_annotations(document):
    images = [r for r in document['images'] if r['split'] == 'test']
    images = sorted(images, key=lambda r: r['filename'])
    captions, records = [], []
    for index, image in enumerate(images):
        image_id = image['filename'].removesuffix('.jpg')
        sentences = image['sentences']
        if set(image['sentids']) != {s['sentid'] for s in sentences}:
            raise ValueError('STOP: inconsistent sentence ownership')
        records.append({'image_id': image_id, 'filename': image['filename'],
                        'karpathy_image_id': image['imgid'], 'caption_indices': []})
        for sentence in sorted(sentences, key=lambda s: s['sentid']):
            if sentence['imgid'] != image['imgid'] or not sentence['raw'].strip():
                raise ValueError('STOP: invalid image-caption ownership')
            records[-1]['caption_indices'].append(len(captions))
            captions.append({'caption_id': str(sentence['sentid']), 'image_id': image_id,
                             'image_index': index, 'caption': sentence['raw']})
    distribution = Counter(len(r['caption_indices']) for r in records)
    if (len(records) != 1000 or len(captions) != 5000 or distribution != {5: 1000}
            or len({r['image_id'] for r in records}) != 1000
            or len({r['karpathy_image_id'] for r in records}) != 1000
            or len({r['caption_id'] for r in captions}) != 5000):
        raise ValueError('STOP: not the expected 1000-image/5000-caption Karpathy test split')
    return records, captions, {'unique_test_images': 1000, 'captions': 5000,
        'captions_per_image_distribution': dict(distribution), 'ownership_validated': True,
        'ordering': 'Images sorted by filename; captions sorted by original Karpathy sentence ID within image.'}


def retrieval_screen(similarities, images, captions):
    scores = np.asarray(similarities)
    owners = np.asarray([c['image_index'] for c in captions])
    if scores.shape != (len(images), len(captions)) or not np.isfinite(scores).all():
        raise ValueError('Invalid full retrieval matrix')
    rows = []
    image_order = np.argsort(-scores, axis=1, kind='stable')
    text_order = np.argsort(-scores.T, axis=1, kind='stable')
    retrieval = {'image_to_text': {}, 'text_to_image': {},
        'ties': 'Stable ordering by saved image/caption index; strict positive margin required for explanation eligibility.'}
    for k in (1, 5, 10):
        retrieval['image_to_text'][f'R@{k}'] = float(np.mean(
            np.any(owners[image_order[:, :k]] == np.arange(len(images))[:, None], axis=1)))
        retrieval['text_to_image'][f'R@{k}'] = float(np.mean(
            np.any(text_order[:, :k] == owners[:, None], axis=1)))
    for i, image in enumerate(images):
        own = np.flatnonzero(owners == i)
        other = np.flatnonzero(owners != i)
        pos = int(own[np.argmax(scores[i, own])])
        neg = int(other[np.argmax(scores[i, other])])
        rows.append({**image, 'positive_caption_index': pos, 'negative_caption_index': neg,
            'positive_caption_id': captions[pos]['caption_id'], 'positive_caption': captions[pos]['caption'],
            'negative_caption_id': captions[neg]['caption_id'], 'negative_caption': captions[neg]['caption'],
            'negative_caption_source_image_id': captions[neg]['image_id'],
            'original_positive_cosine': float(scores[i, pos]), 'original_negative_cosine': float(scores[i, neg]),
            'original_margin': float(scores[i, pos])-float(scores[i, neg])})
    return rows, retrieval


def reserve_order(rows, target=300):
    eligible = sorted([r for r in rows if r['original_margin'] > 0], key=lambda r: r['image_id'])
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate query images')
    if len(eligible) < target:
        raise ValueError(f'STOP: only {len(eligible)} correctly ranked images; need {target}')
    permutation = np.random.default_rng(2026).permutation(len(eligible))
    return [eligible[i] for i in permutation]


def feasible_selection(ordered, feature_counts, target=300):
    """Initial sample fully checked first; replacements follow saved reserve order."""
    initial = ordered[:target]
    exclusions = []
    selected = []
    for position, row in enumerate(ordered):
        if position >= target and len(selected) == target:
            break
        d = feature_counts[row['image_id']]
        if 2 ** d < 128:
            exclusions.append({'image_id': row['image_id'], 'num_superpixels': d,
                'maximum_unique_coalitions': 2 ** d, 'seeded_position': position,
                'reason': 'Technical coalition infeasibility; evaluated before any explanations.'})
        else:
            selected.append(row)
    if len(selected) != target:
        raise ValueError('STOP: insufficient feasible images; do not change settings')
    return initial, selected, exclusions


def paired_statistics(rows, point_key, cr_key, higher_better=False):
    point = np.asarray([r[point_key] for r in rows], dtype=float)
    cr = np.asarray([r[cr_key] for r in rows], dtype=float)
    difference = cr-point if higher_better else point-cr
    if len(point) == 0 or not np.isfinite([point, cr]).all():
        raise ValueError('Invalid paired observations')
    test = wilcoxon(difference, alternative='two-sided', zero_method='wilcox', method='auto') if np.any(difference) else None
    rng = np.random.default_rng(42)
    draws = difference[rng.integers(0, len(rows), size=(10000, len(rows)))].mean(axis=1)
    return {'n': len(rows), 'point_mean': float(point.mean()), 'point_median': float(np.median(point)),
        'cr_mean': float(cr.mean()), 'cr_median': float(np.median(cr)),
        'mean_difference': float(difference.mean()), 'median_difference': float(np.median(difference)),
        'relative_mean_improvement_percent': float(100*difference.mean()/point.mean()) if point.mean() else None,
        'cr_wins': int((difference > 0).sum()), 'point_wins': int((difference < 0).sum()), 'ties': int((difference == 0).sum()),
        'wilcoxon': {'statistic': float(test.statistic) if test else 0., 'pvalue': float(test.pvalue) if test else 1., 'alternative': 'two-sided'},
        'paired_bootstrap_ci95': {'low': float(np.percentile(draws, 2.5)), 'high': float(np.percentile(draws, 97.5)),
                                  'resamples': 10000, 'seed': 42, 'method': 'paired percentile over unique query images'},
        'difference_sign': 'CR minus Pointwise' if higher_better else 'Pointwise minus CR'}


def analyze(rows):
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Paired analysis requires unique query images')
    result = {'primary_endpoint': 'MatchedArea-PDAUC-50',
        'primary': paired_statistics(rows, 'point_matched_pdauc50', 'cr_matched_pdauc50'),
        'secondary_positive_only': paired_statistics(rows, 'point_positive_pdauc', 'cr_positive_pdauc'),
        'budgets': {}, 'exploratory': {},
        'testing': 'Primary predeclared; budget tests and correlations secondary/exploratory, two-sided and unadjusted.'}
    for q in BUDGETS:
        point = np.asarray([r[f'point_rankflip{q}'] for r in rows], dtype=bool)
        cr = np.asarray([r[f'cr_rankflip{q}'] for r in rows], dtype=bool)
        b, c = int((point & ~cr).sum()), int((~point & cr).sum())
        result['budgets'][str(q)] = {
            'preference': paired_statistics(rows, f'point_preference{q}', f'cr_preference{q}'),
            'margin_drop': paired_statistics(rows, f'point_margin_drop{q}', f'cr_margin_drop{q}', True),
            'point_mean_actual_area': float(np.mean([r[f'point_actual_area{q}'] for r in rows])),
            'cr_mean_actual_area': float(np.mean([r[f'cr_actual_area{q}'] for r in rows])),
            'point_rankflip_rate': float(point.mean()), 'cr_rankflip_rate': float(cr.mean()),
            'rankflip_2x2': {'neither_flipped': int((~point & ~cr).sum()), 'point_only_flipped': b,
                             'cr_only_flipped': c, 'both_flipped': int((point & cr).sum())},
            'exact_mcnemar_pvalue': float(binomtest(b, b+c, .5, alternative='two-sided').pvalue) if b+c else 1.}
    difference = [r['matched_pdauc_difference'] for r in rows]
    for key in ('original_margin', 'original_negative_cosine'):
        x = [r[key] for r in rows]
        if len(set(x)) < 2 or len(set(difference)) < 2:
            rho, p = None, None
        else:
            test = spearmanr(x, difference)
            rho, p = float(test.statistic), float(test.pvalue)
        result['exploratory'][key] = {'spearman_rho': rho, 'pvalue': p, 'outcome': 'matched_pdauc_difference'}
    return result
