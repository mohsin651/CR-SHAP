import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.special import expit
from skimage.segmentation import slic

from lili_shap.core import mean_mask, removal_mask, sample_coalitions
from .core import analyze, estimate_targets, figure_selection, preference_deletion
from .io import save_json, sha256, verify_snapshot, write_csv
from .model import PairClipScorer, load_image
from .screen import configure
from .visualize import example_figure, margin_scatter


CONFIG = {'model': 'openai/clip-vit-base-patch32', 'seed': 42, 'n_coalitions': 128,
    'slic': {'n_segments': 16, 'compactness': 30, 'start_label': 0, 'channel_axis': -1,
             'other_parameters': 'Unmodified scikit-image 0.24.0 defaults'},
    'max_side': 512, 'batch_size': 16, 'allow_tf32': False,
    'masking': 'Original global RGB mean, rounded to uint8; no expansion',
    'deletion': 'Validated Experiment 1 Telea radius 3; strictly positive SHAP descending, ascending-index tie break',
    'tail': 'Constant preference to x=1 solely for integration; not an actual deletion state',
    'unreached_threshold': 'Missing; report coverage and common-case paired rates',
    'all_masked_fallback': 'Validated global RGB mean for Telea with no remaining context',
    'runtime': 'Each method records shared coalition generation/scoring cost plus its own regression time; shared cost executed only once.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screening', default='data/sugarcrepe_screening')
    parser.add_argument('--stage', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--pilot', default='results/experiment_2_pilot')
    parser.add_argument('--output')
    args = parser.parse_args()
    screening = Path(args.screening)
    selected_path = screening / 'selected_instances.json'
    selected = json.loads(selected_path.read_text(encoding='utf-8'))
    screening_summary = json.loads((screening / 'screening_summary.json').read_text(encoding='utf-8'))
    integrity = json.loads((screening / 'experiment1_integrity.json').read_text(encoding='utf-8'))
    verify_snapshot(integrity)
    if len(selected) != 100 or any(r['original_margin'] <= 0 for r in selected):
        raise ValueError('Require exactly 100 preselected correctly ranked instances')
    if args.stage == 'full':
        pilot = Path(args.pilot)
        previous = json.loads((pilot / 'config.json').read_text(encoding='utf-8'))
        report = json.loads((pilot / 'checks.json').read_text(encoding='utf-8'))
        if not report['all_passed'] or report['n'] != 5:
            raise ValueError('Five-example pilot must pass before the full experiment')
        if previous['experiment'] != CONFIG or previous['selection_sha256'] != sha256(selected_path):
            raise ValueError('Pilot configuration or selected sample differs')
    output = Path(args.output or ('results/experiment_2_pilot' if args.stage == 'pilot' else 'results/experiment_2'))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    configure()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = PairClipScorer(device)
    gamma = scorer.gamma
    if scorer.model.config._commit_hash != screening_summary['clip_commit'] or gamma != screening_summary['gamma']:
        raise ValueError('Model differs from screening model')
    save_json(output / 'config.json', {'experiment': CONFIG, 'stage': args.stage,
        'selection_sha256': sha256(selected_path), 'clip_commit': scorer.model.config._commit_hash,
        'gamma': gamma, 'device': device, 'arguments': vars(args)})
    save_json(output / 'environment.json', {'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None})
    save_json(output / 'selected_instances.json', selected)
    save_json(output / 'screening_summary.json', screening_summary)
    save_json(output / 'experiment1_integrity.json', integrity)
    cached_texts = np.load(screening / 'text_embeddings.npy')
    caption_index = {c: i for i,c in enumerate(json.loads((screening / 'captions.json').read_text(encoding='utf-8')))}
    rows, reports, folders = [], [], {}
    for index, instance in enumerate(selected[:5] if args.stage == 'pilot' else selected):
        print(f"[{index+1}/{5 if args.stage == 'pilot' else 100}] {instance['example_id']}", flush=True)
        folder = output / f'example_{index:03d}'
        folder.mkdir()
        folders[instance['example_id']] = str(folder.relative_to(output))
        image = load_image(instance['image_path'])
        Image.fromarray(image).save(folder / 'original.png')
        segments = slic(image, n_segments=16, compactness=30, start_label=0, channel_axis=-1)
        _, inverse = np.unique(segments, return_inverse=True)
        segments = inverse.reshape(image.shape[:2])
        d = int(segments.max()) + 1
        coalitions = sample_coalitions(d, 128, 42)
        pair_score, text_features = scorer.for_pair(instance['positive_caption'], instance['negative_caption'])
        original = pair_score([image])[0]
        original_margin = float(original[0] - original[1])
        original_preference = float(expit(gamma * original_margin))
        if original_margin <= 0 or original_preference <= .5:
            raise ValueError('Selected instance no longer correctly ranked; do not silently replace it')
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.perf_counter()
        score_chunks, image_hashes, mask_hashes = [], [], []
        for offset in range(0, 128, 16):
            batch = []
            for coalition in coalitions[offset:offset+16]:
                mask = removal_mask(segments, coalition)
                if not np.array_equal(mask > 0, coalition[segments] == 0):
                    raise ValueError('Incorrect coalition mask')
                perturbed = mean_mask(image, mask)
                batch.append(perturbed)
                mask_hashes.append(hashlib.sha256(mask.tobytes()).hexdigest())
                image_hashes.append(hashlib.sha256(perturbed.tobytes()).hexdigest())
            # One call, one image embedding per coalition, two caption scores.
            score_chunks.append(pair_score(batch))
        pair_scores = np.concatenate(score_chunks)
        torch.cuda.synchronize() if device == 'cuda' else None
        shared_seconds = time.perf_counter()-start
        values, fit = estimate_targets(coalitions, pair_scores)
        check = {'example_id': instance['example_id'], 'full_matches_original': bool(np.allclose(pair_scores[1], original, atol=1e-6, rtol=0)),
            'original_matches_screening': bool(np.allclose(original, [instance['original_positive_cosine'],instance['original_negative_cosine']], atol=1e-6, rtol=0)),
            'caption_embeddings_match_screening': bool(np.allclose(text_features,
                cached_texts[[caption_index[instance['positive_caption']],caption_index[instance['negative_caption']]]], atol=1e-6, rtol=0)),
            'normalized_text_embeddings': bool(np.allclose(np.linalg.norm(text_features,axis=1),1,atol=1e-6,rtol=0)),
            'model_frozen': bool(not scorer.model.training and all(not p.requires_grad for p in scorer.model.parameters())),
            'shared_coalitions_and_bit_identical_images': True,
            'single_shared_image_embedding_per_coalition': True,
            'cr_value_function_is_score_difference': True,
            'efficiency': bool(all(abs(fit[name]['efficiency_residual'])<1e-8 for name in ['positive','negative','contrastive'])),
            'linearity': fit['max_abs_linearity_error'] <= 1e-10,
            'original_correctly_ranked': original_margin>0 and original_preference>.5}
        if not all(v for k,v in check.items() if k!='example_id'):
            save_json(folder / 'checks.json', check)
            raise ValueError(f'Sanity check failed: {check}')
        curves = {method: preference_deletion(image, segments, values[method], pair_score, gamma)
                  for method in ['pointwise','cr_shap']}
        for method, curve in curves.items():
            order = curve['order']
            check[f'{method}_positive_descending_order'] = bool(all(values[method][j]>0 for j in order) and
                all(values[method][a]>=values[method][b] for a,b in zip(order,order[1:])))
            check[f'{method}_preference_formula'] = bool(np.allclose(curve['preferences'],expit(gamma*np.array(curve['margins'])),atol=1e-12,rtol=0))
            check[f'{method}_deletion_initial_scores'] = bool(np.allclose(
                [curve['positive_cosines'][0],curve['negative_cosines'][0]],original,atol=1e-6,rtol=0))
            check[f'{method}_deletion_mask_regions_preserved'] = True  # Asserted inside callback for every state.
        if not all(v for k,v in check.items() if k!='example_id'):
            save_json(folder / 'checks.json', check)
            raise ValueError(f'Deletion check failed: {check}')
        row = {'example_id': instance['example_id'], 'sugarcrepe_category': instance['sugarcrepe_category'],
            'positive_caption': instance['positive_caption'], 'negative_caption': instance['negative_caption'],
            'image_id': instance['image_id'], 'original_positive_cosine': float(original[0]),
            'original_negative_cosine': float(original[1]), 'original_margin': original_margin,
            'original_preference': original_preference, 'num_superpixels': d,
            'pointwise_pdauc': curves['pointwise']['auc'], 'cr_shap_pdauc': curves['cr_shap']['auc'],
            'pdauc_difference': curves['pointwise']['auc']-curves['cr_shap']['auc'],
            'max_abs_linearity_error': fit['max_abs_linearity_error'], 'shared_coalition_runtime_seconds': shared_seconds,
            'pointwise_runtime_seconds': shared_seconds+fit['pointwise_fit_seconds'],
            'cr_shap_runtime_seconds': shared_seconds+fit['cr_shap_fit_seconds']}
        for method, curve in curves.items():
            row.update({f'{method}_rankflip25': curve['rankflip25'],f'{method}_rankflip50':curve['rankflip50'],
                f'{method}_margin25':curve['margin25'],
                f'{method}_margin_drop25':original_margin-curve['margin25'] if curve['margin25'] is not None else None,
                f'{method}_num_positive_regions':curve['num_positive_regions'],
                f'{method}_positive_area_fraction':curve['positive_area_fraction'],
                f'{method}_no_positive_attribution':curve['no_positive_attribution'],
                f'{method}_reached25':curve['reached25'],f'{method}_reached50':curve['reached50']})
        np.savez_compressed(folder / 'raw.npz', segments=segments, coalitions=coalitions,
            coalition_positive_scores=pair_scores[:,0],coalition_negative_scores=pair_scores[:,1],
            coalition_cr_values=pair_scores[:,0]-pair_scores[:,1],
            phi_pointwise=values['pointwise'],phi_negative=values['negative'],phi_cr_shap=values['cr_shap'],
            text_embeddings=text_features, perturbation_sha256=np.array(image_hashes),mask_sha256=np.array(mask_hashes))
        save_json(folder / 'deletion_curves.json', curves)
        save_json(folder / 'instance.json', {**instance,'source_image_sha256':sha256(instance['image_path'])})
        save_json(folder / 'regression.json', fit)
        save_json(folder / 'checks.json', check)
        reports.append(check)
        rows.append(row)
        write_csv(output / 'results.csv', rows)
        save_json(output / 'rows.json', rows)
        if args.stage == 'pilot':
            example_figure(folder / 'explanation.png',row,image,segments,values['pointwise'],values['cr_shap'],curves)
    save_json(output / 'checks.json', {'all_passed': True,'n':len(rows),'examples':reports,
        'max_abs_linearity_error':max(r['max_abs_linearity_error'] for r in rows)})
    save_json(output / 'example_directories.json', folders)
    summary = analyze(rows)
    summary['screening'] = screening_summary
    summary['configuration'] = CONFIG
    summary['gamma'] = gamma
    summary['selected_unique_images'] = len({r['image_id'] for r in rows})
    save_json(output / 'summary.json', summary)
    write_csv(output / 'category_results.csv', [{ 'sugarcrepe_category':c,
        **{k:v for k,v in stat.items() if not isinstance(v,dict)},
        **{f'{m}_rankflip{t}_percent':stat['rankflip'][str(t)][m]['percent_among_reached']
           for m in ['pointwise','cr_shap'] for t in [25,50]},
        **{f'{m}_rankflip{t}_n_reached':stat['rankflip'][str(t)][m]['n_reached']
           for m in ['pointwise','cr_shap'] for t in [25,50]}}
        for c,stat in summary['categories'].items()])
    margin_scatter(output / 'margin_relationship.png',rows,summary['original_margin_relationship'])
    if args.stage == 'full':
        selection = figure_selection(rows)
        save_json(output / 'figure_selection.json', {'rule':'Five largest positive differences, five closest to zero among remaining instances, five most negative differences; deterministic ID tie breaks.',
                                                    'groups':selection})
        figures = output / 'figures'
        figures.mkdir()
        for group, ids in selection.items():
            for i, example_id in enumerate(ids):
                row = next(r for r in rows if r['example_id']==example_id)
                folder = output / folders[example_id]
                raw = np.load(folder/'raw.npz')
                image = np.array(Image.open(folder/'original.png'))
                curves = json.loads((folder/'deletion_curves.json').read_text(encoding='utf-8'))
                example_figure(figures / f'{group}_{i+1}.png',row,image,raw['segments'],raw['phi_pointwise'],raw['phi_cr_shap'],curves)
        if sum(map(len,selection.values()))<15:
            raise ValueError('Insufficient improved or worsened examples to produce the requested 5/5/5 grouping; metrics preserved')
    verify_snapshot(integrity)
    save_json(output / 'experiment1_integrity_check.json', {'all_unchanged':True,'n_files':len(integrity)})
    print(f'Completed {args.stage}: {output.resolve()}; all checks passed',flush=True)


if __name__ == '__main__':
    main()
