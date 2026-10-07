"""Full zero-shot retrieval screening, then frozen sample feasibility preflight."""
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from skimage.segmentation import slic
import torch

from experiment2.io import save_json, sha256, verify_snapshot, write_csv
from experiment2.model import load_image
from experiment2.screen import configure
from experiment3.run import COMMIT, PinnedScorer
from lili_shap.core import sample_coalitions
from .core import feasible_selection, reserve_order, retrieval_screen
from .data import ROOT

SCREEN = Path('data/flickr30k_screening_exp4')


def protected_snapshot():
    paths = []
    for directory in ('lili_shap', 'experiment2', 'experiment3', 'tests'):
        paths.extend(Path(directory).glob('*.py'))
    paths.extend(Path('.').glob('*.md'))
    paths.extend(Path('.').glob('requirements*.txt'))
    paths.extend(p for p in Path('results').rglob('*') if p.is_file()
                 and not any(part.startswith('experiment_4') for part in p.parts))
    return {str(p): sha256(p) for p in sorted(paths)}


def segment(image):
    labels = slic(image, n_segments=16, compactness=30, start_label=0, channel_axis=-1)
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(image.shape[:2])


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'median': float(np.median(a)), 'standard_deviation': float(a.std()),
            'minimum': float(a.min()), 'maximum': float(a.max()), 'std_ddof': 0}


def main():
    if SCREEN.exists():
        raise FileExistsError(SCREEN)
    validation = json.loads((ROOT/'split_validation.json').read_text())
    if validation['unique_test_images'] != 1000 or validation['captions'] != 5000 or not validation['ownership_validated']:
        raise ValueError('STOP: invalid split')
    images = json.loads((ROOT/'images.json').read_text())
    captions = json.loads((ROOT/'captions.json').read_text())
    if importlib.metadata.version('scikit-image') != '0.24.0':
        raise ValueError('STOP: incorrect SLIC version')
    for row in images:
        if sha256(row['image_path']) != row['sha256']:
            raise ValueError('STOP: image hash mismatch')
    SCREEN.mkdir()
    print('Snapshotting prior experiments', flush=True)
    protected = protected_snapshot()
    save_json(SCREEN/'protected_experiments.json', protected)
    configure()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = PinnedScorer(device)
    if scorer.model.config._commit_hash != COMMIT or abs(scorer.gamma-100.) > 1e-4:
        raise ValueError('STOP: model or gamma mismatch')
    started = time.perf_counter()
    text = scorer.encode_texts([r['caption'] for r in captions])
    features = []
    for offset in range(0, len(images), 16):
        features.append(scorer.encode_images([load_image(r['image_path']) for r in images[offset:offset+16]]))
        if offset % 160 == 0:
            print(f'Screened {min(offset+16,len(images))}/1000 images', flush=True)
    features = np.concatenate(features)
    if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-6) or not np.allclose(np.linalg.norm(text, axis=1), 1, atol=1e-6):
        raise ValueError('STOP: non-normalized embeddings')
    matrix = features @ text.T
    rows, retrieval = retrieval_screen(matrix, images, captions)
    for row in rows:
        row['original_preference'] = float(expit(scorer.gamma*row['original_margin']))
        row['correctly_ranked'] = row['original_margin'] > 0
    np.save(SCREEN/'image_embeddings.npy', features)
    np.save(SCREEN/'text_embeddings.npy', text)
    np.save(SCREEN/'similarities.npy', matrix)
    save_json(SCREEN/'images.json', images)
    save_json(SCREEN/'captions.json', captions)
    save_json(SCREEN/'rows.json', rows)
    write_csv(SCREEN/'screening.csv', rows)
    summary = {'total_test_images': 1000, 'total_captions': 5000,
        'correctly_ranked': sum(r['correctly_ranked'] for r in rows),
        'incorrect_or_tied': sum(not r['correctly_ranked'] for r in rows),
        'ties': sum(r['original_margin'] == 0 for r in rows),
        'correctly_ranked_percent': 100*np.mean([r['correctly_ranked'] for r in rows]),
        'distributions': {key: distribution([r[key] for r in rows]) for key in
            ('original_positive_cosine', 'original_negative_cosine', 'original_margin')},
        'retrieval': retrieval, 'clip_commit': COMMIT, 'gamma': scorer.gamma,
        'screening_runtime_seconds': time.perf_counter()-started, 'device': device,
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'dataset_source': json.loads((ROOT/'source.json').read_text()), 'split_validation': validation}
    save_json(SCREEN/'screening_summary.json', summary)
    ordered = reserve_order(rows)
    save_json(SCREEN/'seeded_sample_and_reserve.json', ordered)
    save_json(SCREEN/'initial_selected_images.json', ordered[:300])
    preflight_dir = SCREEN/'preflight'
    preflight_dir.mkdir()
    counts, checks = {}, []

    def check(row, position):
        image = load_image(row['image_path'])
        labels = segment(image)
        d = int(labels.max())+1
        counts[row['image_id']] = d
        np.save(preflight_dir/f"{row['image_id']}.npy", labels)
        feasible = 2 ** d >= 128
        if feasible:
            sample_coalitions(d, 128, 42)  # Also verifies singleton/complement constraints.
        checks.append({'image_id': row['image_id'], 'seeded_position': position, 'num_superpixels': d,
            'maximum_unique_coalitions': 2 ** d, 'feasible': feasible,
            'segmentation_sha256': sha256(preflight_dir/f"{row['image_id']}.npy"),
            'working_image_sha256': __import__('hashlib').sha256(image.tobytes()).hexdigest()})

    for i, row in enumerate(ordered[:300]):
        check(row, i)
        if (i+1) % 50 == 0:
            print(f'Preflight checked {i+1}/300 initial images; no explanations computed', flush=True)
    position = 300
    while sum(v >= 7 for v in counts.values()) < 300:
        if position == len(ordered):
            raise ValueError('STOP: reserve exhausted')
        check(ordered[position], position)
        position += 1
    initial, selected, exclusions = feasible_selection(ordered[:position], counts)
    save_json(SCREEN/'technical_exclusions.json', exclusions)
    save_json(SCREEN/'selected_images.json', selected)
    save_json(SCREEN/'segmentation_preflight.json', {'all_passed': True, 'n_final': 300, 'checks': checks,
        'timing': 'All initial 300 checked, then seeded reserves; before any SHAP/deletion outcomes.'})
    save_json(SCREEN/'selection_report.json', {'seed': 2026, 'eligible_unique_images': len(ordered),
        'initial_n': len(initial), 'final_n': len(selected), 'unique_images': len({r['image_id'] for r in selected}),
        'technical_exclusions': len(exclusions), 'reserve_images_checked': position-300,
        'replacement_ids': [r['image_id'] for r in selected if r not in initial],
        'rule': 'Uniform seeded permutation over correctly ranked unique images; technical replacements from fixed reserve only before outcomes.',
        'selection_sha256': sha256(SCREEN/'selected_images.json')})
    save_json(SCREEN/'screening_artifact_hashes.json', {str(p.relative_to(SCREEN)): sha256(p)
        for p in SCREEN.iterdir() if p.is_file()})
    verify_snapshot(protected)
    print(f"Screening and frozen selection complete: {summary['correctly_ranked']} eligible, 300 selected, {len(exclusions)} technical exclusions", flush=True)


if __name__ == '__main__':
    main()
