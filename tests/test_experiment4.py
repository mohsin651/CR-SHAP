import numpy as np
import pytest

from experiment4.core import feasible_selection, paired_statistics, reserve_order, retrieval_screen, validate_annotations


def test_retrieval_ownership_duplicate_text_and_full_pool_winners():
    images = [{'image_id': str(i)} for i in range(3)]
    captions = [{'caption_id': str(j), 'image_id': str(j//2), 'image_index': j//2,
                 'caption': 'same text' if j in (0, 2) else str(j)} for j in range(6)]
    scores = np.array([[.8, .9, .95, .2, .1, .3], [.1, .2, .7, .8, .3, .4], [.1, .3, .2, .4, .9, .8]])
    rows, recall = retrieval_screen(scores, images, captions)
    assert rows[0]['positive_caption_id'] == '1'
    assert rows[0]['negative_caption_id'] == '2'
    assert rows[0]['negative_caption_source_image_id'] == '1'
    assert rows[0]['original_margin'] < 0
    assert all(r['negative_caption_source_image_id'] != r['image_id'] for r in rows)
    assert recall['image_to_text']['R@1'] == 2/3
    assert recall['text_to_image']['R@1'] == 5/6
    assert recall['image_to_text']['R@5'] == 1


def test_seeded_reserve_feasibility_exclusion_before_outcomes():
    rows = [{'image_id': str(i), 'original_margin': 1. if i < 8 else -1.} for i in range(10)]
    ordered = reserve_order(rows, target=3)
    assert ordered == reserve_order(list(reversed(rows)), target=3)
    counts = {r['image_id']: 16 for r in ordered}
    counts[ordered[1]['image_id']] = 6
    initial, selected, exclusions = feasible_selection(ordered, counts, target=3)
    assert initial == ordered[:3]
    assert selected == [ordered[0], ordered[2], ordered[3]]
    assert exclusions[0]['maximum_unique_coalitions'] == 64
    with pytest.raises(ValueError, match='only 8'):
        reserve_order(rows, target=9)


def test_paired_inference_sign_and_exact_ties():
    rows = [{'p': 1., 'c': .5} for _ in range(4)]
    stats = paired_statistics(rows, 'p', 'c')
    assert stats['mean_difference'] == .5 and stats['cr_wins'] == 4
    assert stats['relative_mean_improvement_percent'] == 50.
    assert stats['paired_bootstrap_ci95']['low'] == stats['paired_bootstrap_ci95']['high'] == .5
    assert paired_statistics(rows, 'p', 'c', higher_better=True)['cr_wins'] == 0
    ties = paired_statistics([{'p': .5, 'c': .5}], 'p', 'c')
    assert ties['ties'] == 1 and ties['wilcoxon']['pvalue'] == 1.


def test_reject_nonstandard_split_and_inconsistent_ownership():
    document = {'images': [{'split': 'test', 'filename': 'one.jpg', 'imgid': 1,
        'sentids': [1], 'sentences': [{'sentid': 1, 'imgid': 1, 'raw': 'caption'}]}]}
    with pytest.raises(ValueError, match='expected 1000-image'):
        validate_annotations(document)
    document['images'][0]['sentences'][0]['imgid'] = 2
    with pytest.raises(ValueError, match='ownership'):
        validate_annotations(document)
