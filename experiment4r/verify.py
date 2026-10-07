"""Independently reconstruct saved retrieval and explanation evidence, no CLIP rerun."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit

from experiment2.io import save_json, sha256, verify_snapshot
from lili_shap.core import kernel_shap, mean_mask, positive_order, removal_mask, sample_coalitions
from .core import BUDGETS, analyze, retrieval_screen
from .run import CONFIG, source_hashes
from experiment4.screen import SCREEN
from .prepare import INPUT,ROOT


def require(condition, message):
    if not condition:
        raise ValueError(f'STOP: saved-evidence validation failed: {message}')


def verify(root):
    config = json.loads((root/'config.json').read_text())
    require(config['experiment'] == CONFIG and config['source_hashes'] == source_hashes(), 'method/source mismatch')
    require(config['selection_sha256'] == sha256(INPUT/'selected_images.json'), 'selection hash')
    for name, digest in json.loads((root/'screening_artifact_hashes.json').read_text()).items():
        require(sha256(SCREEN/name) == digest, f'screening hash: {name}')
    gamma = config['gamma']
    images = json.loads((SCREEN/'images.json').read_text())
    captions = json.loads((SCREEN/'captions.json').read_text())
    matrix = np.load(SCREEN/'similarities.npy')
    features, texts = np.load(SCREEN/'image_embeddings.npy'), np.load(SCREEN/'text_embeddings.npy')
    require(np.array_equal(features @ texts.T, matrix), 'full retrieval matrix from embeddings')
    screening_rows, retrieval = retrieval_screen(matrix, images, captions)
    require(retrieval == json.loads((SCREEN/'screening_summary.json').read_text())['retrieval'], 'retrieval metrics')
    screen_saved = json.loads((SCREEN/'rows.json').read_text())
    for row, saved in zip(screening_rows, screen_saved):
        require(all(saved[k] == v for k, v in row.items()), 'full-pool positive/strongest false caption')
    selected = json.loads((INPUT/'selected_images.json').read_text())
    require(selected == json.loads((root/'selected_images.json').read_text()), 'frozen remainder selection')
    failures = json.loads((root/'technical_failures.json').read_text())
    failed_ids = {r['image_id'] for r in failures}
    expected = selected[:5] if config['stage'] == 'pilot' else selected
    expected = [r for r in expected if r['image_id'] not in failed_ids]
    rows = json.loads((root/'rows.json').read_text())
    require([r['image_id'] for r in rows] == [r['image_id'] for r in expected], 'all evaluable remainder, no replacement')
    integrity = json.loads((root/'model_integrity.json').read_text())
    require(integrity['before_sha256'] == integrity['after_sha256'], 'unchanged model weights')
    positions = {r['image_id']:i for i,r in enumerate(selected)}
    require(json.loads((root/'checks.json').read_text())['all_passed'], 'runner checks')
    max_error = 0.
    for index, row in enumerate(rows):
        selected_index = positions[row['image_id']]
        folder = root/f'example_{selected_index:03d}'
        instance = json.loads((folder/'instance.json').read_text())
        require(instance == selected[selected_index], 'instance identity')
        require(row['negative_caption_source_image_id'] != row['image_id'], 'negative ownership')
        require(row['positive_caption_id'] == instance['positive_caption_id'] and row['negative_caption_id'] == instance['negative_caption_id'], 'caption IDs')
        require(row['positive_caption'] == instance['positive_caption'] and row['negative_caption'] == instance['negative_caption'], 'caption texts')
        image = np.asarray(Image.open(folder/'original.png'))
        with np.load(folder/'raw.npz') as raw:
            segments, coalitions = raw['segments'], raw['coalitions']
            d = row['num_superpixels']
            require(np.array_equal(coalitions, sample_coalitions(d, 128, 42)), 'unchanged unique coalition sampler')
            require(np.array_equal(segments, np.load(INPUT/'preflight'/f"{row['image_id']}.npy")), 'preflight segmentation')
            require(np.array_equal(np.unique(segments), np.arange(d)), 'contiguous segmentation')
            for i, coalition in enumerate(coalitions):
                mask = removal_mask(segments, coalition)
                require(hashlib.sha256(mask.tobytes()).hexdigest() == raw['mask_sha256'][i], 'coalition pixel-mask hash')
                require(hashlib.sha256(mean_mask(image, mask).tobytes()).hexdigest() == raw['perturbation_sha256'][i], 'coalition perturbation hash')
            pairs = np.column_stack([raw['coalition_positive_scores'], raw['coalition_negative_scores']])
            require(np.allclose(raw['coalition_image_embeddings'] @ raw['text_embeddings'].T, pairs, atol=1e-7, rtol=0), 'shared embedding cosine scores')
            require(np.allclose(np.linalg.norm(raw['coalition_image_embeddings'], axis=1), 1, atol=1e-6), 'normalized image embeddings')
            require(np.allclose(raw['text_embeddings'], texts[[instance['positive_caption_index'], instance['negative_caption_index']]], atol=1e-6, rtol=0), 'cached caption vectors')
            for key, scores in (('phi_pointwise', pairs[:, 0]), ('phi_negative', pairs[:, 1]), ('phi_cr_shap', pairs[:, 0]-pairs[:, 1])):
                phi, _ = kernel_shap(coalitions, scores)
                require(np.allclose(phi, raw[key], atol=1e-12, rtol=0), 'constrained regression reconstruction')
            error = float(np.max(abs(raw['phi_cr_shap']-raw['phi_pointwise']+raw['phi_negative'])))
            require(error < 1e-10 and np.isclose(error, row['max_abs_linearity_error'], atol=1e-15), 'Shapley linearity')
            max_error = max(max_error, error)
            original = [row['original_positive_cosine'], row['original_negative_cosine']]
            require(np.array_equal(pairs[1], original), 'original/full coalition')
            require(np.allclose(original, [instance['original_positive_cosine'], instance['original_negative_cosine']], atol=1e-6, rtol=0), 'screening score match')
            require(row['original_margin'] > 0 and row['original_preference'] > .5, 'strict original eligibility')
            require(np.isclose(row['original_margin'], original[0]-original[1], atol=1e-12), 'original margin')
            require(np.isclose(row['original_preference'], expit(gamma*row['original_margin']), atol=1e-12), 'original preference')
            positive = json.loads((folder/'positive_deletion.json').read_text())
            matched = json.loads((folder/'matched_deletion.json').read_text())
            for prefix, phi_key in (('point', 'phi_pointwise'), ('cr', 'phi_cr_shap')):
                phi, curve, full = raw[phi_key], positive[prefix], matched[prefix]
                require(curve['order'] == positive_order(phi).tolist(), 'strictly positive ranking')
                mask = np.zeros(segments.shape, dtype=bool)
                fractions = [0.]
                for j in curve['order']:
                    mask |= segments == j
                    fractions.append(float(mask.mean()))
                require(fractions == curve['fractions'], 'positive actual fractions')
                for c in (curve, full):
                    margin = np.asarray(c['positive_cosines'])-np.asarray(c['negative_cosines'])
                    require(np.allclose(margin, c['margins'], atol=1e-12, rtol=0), 'deletion margins')
                    require(np.allclose(expit(gamma*margin), c['preferences'], atol=1e-12, rtol=0), 'deletion preference')
                tail_x, tail_y = fractions.copy(), curve['preferences'].copy()
                if tail_x[-1] < 1:
                    tail_x.append(1.)
                    tail_y.append(tail_y[-1])
                require(tail_x == curve['auc_fractions'] and tail_y == curve['auc_scores'], 'constant integration-only tail')
                require(np.isclose(np.trapz(tail_y, tail_x), row[f'{prefix}_positive_pdauc'], atol=1e-12), 'positive PDAUC')
                require(row[f'{prefix}_num_positive_regions'] == len(curve['order']) and row[f'{prefix}_positive_area_fraction'] == fractions[-1], 'positive coverage')
                require(row[f'{prefix}_final_positive_preference'] == curve['preferences'][-1], 'final positive preference')
                require(full['order'] == sorted(range(d), key=lambda j: (-phi[j], j)), 'all signed ranking')
                masks = raw[f'{prefix}_matched_masks']
                require(masks.shape == (d+1, *segments.shape), 'saved mask shape')
                mask[:] = False
                for state in range(d+1):
                    if state:
                        mask |= segments == full['order'][state-1]
                    require(np.array_equal(masks[state] > 0, mask), 'cumulative whole-region mask')
                    require(full['fractions'][state] == float(mask.mean()), 'matched actual area')
                states = [0]
                for q in BUDGETS:
                    state = full['budget_states'][str(q)]
                    require(full['fractions'][state-1] < q/100 <= full['fractions'][state], 'first budget crossing')
                    states.append(state)
                    for field, curve_key in (('actual_area', 'fractions'), ('preference', 'preferences'), ('margin', 'margins'),
                                             ('positive_cosine', 'positive_cosines'), ('negative_cosine', 'negative_cosines')):
                        require(row[f'{prefix}_{field}{q}'] == full[curve_key][state], f'budget {field}')
                    require(row[f'{prefix}_rankflip{q}'] == (full['margins'][state] < 0), 'rank flip')
                    require(np.isclose(row[f'{prefix}_margin_drop{q}'], row['original_margin']-full['margins'][state], atol=1e-12), 'margin drop')
                x, y = [full['fractions'][s] for s in states], [full['preferences'][s] for s in states]
                require(x == full['auc_fractions'] and y == full['auc_preferences'], 'six matched AUC states')
                require(np.isclose(np.trapz(y, x)/x[-1], row[f'{prefix}_matched_pdauc50'], atol=1e-12), 'normalized matched PDAUC')
            require(np.isclose(row['matched_pdauc_difference'], row['point_matched_pdauc50']-row['cr_matched_pdauc50'], atol=1e-12), 'primary difference sign')
            require(np.isclose(row['positive_pdauc_difference'], row['point_positive_pdauc']-row['cr_positive_pdauc'], atol=1e-12), 'secondary difference sign')
        if (index+1) % 50 == 0:
            print(f'Validated {index+1}/{len(rows)} saved examples', flush=True)
    require(analyze(rows) == json.loads((root/'summary.json').read_text()), 'recomputed paired inference')
    protected = json.loads((root/'protected_experiments.json').read_text())
    verify_snapshot(protected)
    save_json(root/'verification.json', {'all_passed': True, 'n': len(rows), 'retrieval_images_validated': 1000,
        'retrieval_captions_validated': 5000, 'shared_coalitions_validated': 128*len(rows),
        'max_abs_linearity_error': max_error, 'protected_files_unchanged': len(protected),
        'scope': 'Full retrieval winners/recall, frozen reserve selection, coalition pixel hashes and shared embeddings, regressions, signed/positive rankings, masks, budget crossings, formulas and statistics. No new model inference.'})
    print('Independent saved-evidence verification passed', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', default=str(ROOT/'new_full'))
    verify(Path(parser.parse_args().directory))


if __name__ == '__main__':
    main()
