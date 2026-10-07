"""Frozen shared-inference explanation run; no outcome-based replacement."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from scipy.special import expit
import torch

from experiment2.core import estimate_targets, preference_deletion
from experiment2.io import save_json, sha256, verify_snapshot, write_csv
from experiment2.model import load_image
from experiment2.screen import configure
from experiment3.run import CONFIG as EXP3_CONFIG
from .model import COMMIT, MODEL, PinnedScorer, model_digest
from lili_shap.core import mean_mask, removal_mask, sample_coalitions
from .core import BUDGETS, analyze, matched_deletion
from .screen import root_for, segment

CONFIG = {**EXP3_CONFIG, 'model': MODEL, 'clip_commit': COMMIT,
    'dataset': 'Separate SugarCrepe and Flickr30k B/16 replication arms',
    'selection_seed': 2027, 'selection': '200 unique images per arm; uniform image permutation, one eligible instance per image, fixed reserves before outcomes',
    'primary_endpoint': 'MatchedArea-PDAUC-50', 'secondary_endpoint': 'PositiveOnly-PDAUC',
    'bootstrap': {'resamples':10000,'seed':42,'unit':'unique image'},
    'captions': 'SugarCrepe dataset-defined pairs; Flickr30k B/16 highest GT and strongest other-image caption in full5000 pool',
    'model_geometry': {'input':224,'patch':16,'grid':[14,14],'tokens':197,'SLIC_features_unchanged':True}}



def source_hashes():
    return {str(p): sha256(p) for p in sorted(Path('experiment7').glob('*.py'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--arm', choices=('sugarcrepe','flickr30k'), required=True)
    args = parser.parse_args()
    SCREEN = root_for(args.arm)
    output = Path('results/experiment_7')/args.arm/args.stage
    if output.exists():
        raise FileExistsError(output)
    for name, digest in json.loads((SCREEN/'screening_artifact_hashes.json').read_text()).items():
        if sha256(SCREEN/name) != digest:
            raise ValueError(f'STOP: screening artifact changed: {name}')
    preflight = json.loads((SCREEN/'segmentation_preflight.json').read_text())
    selected = json.loads((SCREEN/'selected_images.json').read_text())
    if not preflight['all_passed'] or len(selected) != 200 or len({r['image_id'] for r in selected}) != 200:
        raise ValueError('STOP: frozen selection/preflight invalid')
    protected = json.loads((SCREEN/'protected_experiments.json').read_text())
    verify_snapshot(protected)
    if importlib.metadata.version('scikit-image') != '0.24.0':
        raise ValueError('STOP: SLIC version mismatch')
    pilot = Path('results/experiment_7')/args.arm/'pilot'
    if args.stage == 'full':
        previous = json.loads((pilot/'config.json').read_text())
        if (previous['experiment'] != CONFIG or previous['selection_sha256'] != sha256(SCREEN/'selected_images.json')
                or previous['source_hashes'] != source_hashes()
                or not json.loads((pilot/'checks.json').read_text())['all_passed']
                or not json.loads((pilot/'verification.json').read_text())['all_passed']):
            raise ValueError('STOP: independently verified matching pilot required')
    output.mkdir(parents=True)
    configure()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = PinnedScorer(device)
    gamma = scorer.gamma
    before = model_digest(scorer.model)
    if scorer.model.config.vision_config.image_size != 224 or scorer.model.config.vision_config.patch_size != 16:
        raise ValueError('Backbone geometry mismatch')
    if scorer.model.config._commit_hash != COMMIT or not (np.isfinite(gamma) and gamma > 0):
        raise ValueError('STOP: model/gamma mismatch')
    save_json(output/'config.json', {'experiment': CONFIG, 'gamma': gamma, 'stage': args.stage,
        'selection_sha256': sha256(SCREEN/'selected_images.json'), 'source_hashes': source_hashes(), 'arm': args.arm})
    save_json(output/'environment.json', {'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None})
    for name in ('screening_summary.json', 'selection_report.json', 'protected_experiments.json',
                 'technical_exclusions.json', 'segmentation_preflight.json', 'screening_artifact_hashes.json'):
        shutil.copy2(SCREEN/name, output/name)
    save_json(output/'selected_images.json', selected)
    text_cache = np.load(SCREEN/'text_embeddings.npy')
    preflight_by_id = {r['image_id']: r for r in preflight['checks']}
    sample = selected[:5] if args.stage == 'pilot' else selected
    rows, checks = [], []
    if args.stage == 'full':
        rows = json.loads((pilot/'rows.json').read_text())
        if [r['image_id'] for r in rows] != [r['image_id'] for r in selected[:5]]:
            raise ValueError('STOP: pilot prefix differs')
        checks = json.loads((pilot/'checks.json').read_text())['examples']
        for i in range(len(rows)):
            shutil.copytree(pilot/f'example_{i:03d}', output/f'example_{i:03d}')
        save_json(output/'reused_pilot.json', {'source': str(pilot), 'n': len(rows),
            'reason': 'Identical frozen method, source hashes and selected prefix; independently verified evidence reused verbatim.'})
    reused = len(rows)
    for index, instance in enumerate(sample):
        if index < reused:
            continue
        print(f'[{index+1}/{len(sample)}] image {instance["image_id"]}', flush=True)
        folder = output/f'example_{index:03d}'
        folder.mkdir()
        if sha256(instance['image_path']) != instance['sha256']:
            raise ValueError('STOP: source image changed')
        image = load_image(instance['image_path'])
        labels_path = SCREEN/'preflight'/f"{instance['image_id']}.npy"
        evidence = preflight_by_id[instance['image_id']]
        if (sha256(labels_path) != evidence['segmentation_sha256']
                or hashlib.sha256(image.tobytes()).hexdigest() != evidence['working_image_sha256']):
            raise ValueError('STOP: preflight image or segmentation changed')
        segments = np.load(labels_path)
        if not np.array_equal(segments, segment(image)):
            raise ValueError('STOP: frozen segmentation no longer matches')
        Image.fromarray(image).save(folder/'original.png')
        d = int(segments.max())+1
        coalitions = sample_coalitions(d, 128, 42)
        text = scorer.encode_texts([instance['positive_caption'], instance['negative_caption']])
        expected_text = text_cache[[instance['positive_caption_index'], instance['negative_caption_index']]]
        if not np.allclose(text, expected_text, atol=1e-6, rtol=0):
            raise ValueError('STOP: cached caption embeddings differ')
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        embeddings, image_hashes, mask_hashes = [], [], []
        for offset in range(0, 128, 16):
            batch = []
            for coalition in coalitions[offset:offset+16]:
                mask = removal_mask(segments, coalition)
                perturbed = mean_mask(image, mask)
                batch.append(perturbed)
                image_hashes.append(hashlib.sha256(perturbed.tobytes()).hexdigest())
                mask_hashes.append(hashlib.sha256(mask.tobytes()).hexdigest())
            embeddings.append(scorer.encode_images(batch))
        embeddings = np.concatenate(embeddings)
        pairs = (embeddings @ text.T).astype(float)
        if device == 'cuda':
            torch.cuda.synchronize()
        shared_seconds = time.perf_counter()-started
        original = pairs[1]
        margin = float(original[0]-original[1])
        values, fit = estimate_targets(coalitions, pairs)
        check = {'image_id': instance['image_id'], 'positive_original_margin': margin > 0,
            'positive_original_preference': float(expit(gamma*margin)) > .5,
            'screening_matches': bool(np.allclose(original, [instance['original_positive_cosine'], instance['original_negative_cosine']], atol=1e-6, rtol=0)),
            'caption_definition_valid': args.arm == 'sugarcrepe' or instance['negative_caption_source_image_id'] != instance['image_id'],
            'frozen': not scorer.model.training and all(not p.requires_grad for p in scorer.model.parameters()),
            'efficiency': all(abs(fit[k]['efficiency_residual']) < 1e-8 for k in ('positive', 'negative', 'contrastive')),
            'linearity': fit['max_abs_linearity_error'] < 1e-10}
        if not all(v for k, v in check.items() if k != 'image_id'):
            save_json(folder/'checks.json', check)
            raise ValueError(f'STOP: {check}')
        # Cache identical evaluator states across both methods/endpoints. Coalition
        # embeddings remain saved once and both targets share each image encoding.
        evaluation_cache = {hashlib.sha256(image.tobytes()).hexdigest(): original}
        def score(images):
            keys = [hashlib.sha256(x.tobytes()).hexdigest() for x in images]
            missing = {}
            for key, x in zip(keys, images):
                if key not in evaluation_cache:
                    missing.setdefault(key, x)
            if missing:
                predictions = (scorer.encode_images(list(missing.values())) @ text.T).astype(float)
                evaluation_cache.update(zip(missing, predictions))
            return np.asarray([evaluation_cache[key] for key in keys])
        positive, matched, masks = {}, {}, {}
        row = {key: instance[key] for key in ('image_id', 'positive_caption_id', 'positive_caption',
            'negative_caption_id', 'negative_caption', 'negative_caption_source_image_id')}
        row.update({'original_positive_cosine': float(original[0]), 'original_negative_cosine': float(original[1]),
            'original_margin': margin, 'original_preference': float(expit(gamma*margin)), 'num_superpixels': d,
            'max_abs_linearity_error': fit['max_abs_linearity_error'], 'shared_inference_runtime_seconds': shared_seconds,
            'point_regression_runtime_seconds': fit['pointwise_fit_seconds'], 'cr_regression_runtime_seconds': fit['cr_shap_fit_seconds']})
        evaluation_start = time.perf_counter()
        for prefix, method in (('point', 'pointwise'), ('cr', 'cr_shap')):
            positive[prefix] = preference_deletion(image, segments, values[method], score, gamma)
            matched[prefix], masks[prefix] = matched_deletion(image, segments, values[method], score, gamma)
            curve, full = positive[prefix], matched[prefix]
            for q in BUDGETS:
                state = full['budget_states'][str(q)]
                check[f'{prefix}_crossing{q}'] = full['fractions'][state-1] < q/100 <= full['fractions'][state]
                row.update({f'{prefix}_preference{q}': full['preferences'][state], f'{prefix}_margin{q}': full['margins'][state],
                    f'{prefix}_positive_cosine{q}': full['positive_cosines'][state], f'{prefix}_negative_cosine{q}': full['negative_cosines'][state],
                    f'{prefix}_margin_drop{q}': margin-full['margins'][state], f'{prefix}_rankflip{q}': full['margins'][state] < 0,
                    f'{prefix}_actual_area{q}': full['fractions'][state]})
            row.update({f'{prefix}_positive_pdauc': curve['auc'], f'{prefix}_positive_area_fraction': curve['positive_area_fraction'],
                f'{prefix}_num_positive_regions': curve['num_positive_regions'], f'{prefix}_no_positive_attribution': curve['no_positive_attribution'],
                f'{prefix}_final_positive_preference': curve['preferences'][-1], f'{prefix}_matched_pdauc50': full['matched_pdauc50']})
        row['evaluation_runtime_seconds'] = time.perf_counter()-evaluation_start
        row['positive_pdauc_difference'] = row['point_positive_pdauc']-row['cr_positive_pdauc']
        row['matched_pdauc_difference'] = row['point_matched_pdauc50']-row['cr_matched_pdauc50']
        if not all(v for k, v in check.items() if k != 'image_id'):
            raise ValueError(f'STOP: {check}')
        np.savez_compressed(folder/'raw.npz', segments=segments, coalitions=coalitions,
            coalition_image_embeddings=embeddings, coalition_positive_scores=pairs[:, 0], coalition_negative_scores=pairs[:, 1],
            phi_pointwise=values['pointwise'], phi_negative=values['negative'], phi_cr_shap=values['cr_shap'],
            text_embeddings=text, perturbation_sha256=np.asarray(image_hashes), mask_sha256=np.asarray(mask_hashes),
            point_matched_masks=masks['point'], cr_matched_masks=masks['cr'])
        save_json(folder/'positive_deletion.json', positive)
        save_json(folder/'matched_deletion.json', matched)
        save_json(folder/'instance.json', instance)
        save_json(folder/'regression.json', fit)
        save_json(folder/'checks.json', check)
        rows.append(row)
        checks.append(check)
        write_csv(output/'results.csv', rows)
        save_json(output/'rows.json', rows)
    after = model_digest(scorer.model)
    if after != before: raise ValueError('Model state changed')
    save_json(output/'model_integrity.json', {'before_sha256':before,'after_sha256':after,'unchanged':True})
    save_json(output/'checks.json', {'all_passed': True, 'n': len(rows), 'examples': checks,
        'max_abs_linearity_error': max(r['max_abs_linearity_error'] for r in rows)})
    save_json(output/'summary.json', analyze(rows))
    verify_snapshot(protected)
    save_json(output/'integrity_check.json', {'all_unchanged': True, 'n_files': len(protected)})
    print(f'Completed {output}; all checks passed', flush=True)


if __name__ == '__main__':
    main()
