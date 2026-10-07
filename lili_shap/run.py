import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from skimage.segmentation import slic

from .core import METHODS, deletion_curve, kernel_shap, mean_mask, removal_mask, sample_coalitions, summarize
from .data import load_pairs


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Frozen CLIP / LILI-style Kernel SHAP prototype')
    parser.add_argument('--manifest', default='data/flickr30k/pairs.csv')
    parser.add_argument('--output', default='results')
    parser.add_argument('--stage', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--reviewed-pilot', help='Pilot directory with review.json marking visual checks passed')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--lama-weights')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-side', type=int, default=512)
    args = parser.parse_args()
    import torch
    from .models import ClipScorer, LamaInpainter
    from .visualize import explanation_figure, perturbation_figure
    pairs = load_pairs(args.manifest)
    if len(pairs) < 30:
        raise ValueError('Prepare 30 unique matched image-caption pairs, even for the pilot')
    # Downloader has already sampled with seed 42; preserve its order so the
    # pilot is exactly the first three images of the full run.
    config = {'seed': 42, 'n_segments': 16, 'n_coalitions': 128, 'dilation_radius': 3,
              'slic_compactness': 30, 'telea_radius': 3, 'model': 'openai/clip-vit-base-patch32',
              'max_side': args.max_side, 'batch_size': args.batch_size,
              'allow_tf32': False, 'lama_jit_optimization': False,
              'auc': 'raw cosine, pixel-area x axis, constant tail to 1 after positive-only deletion',
              'all_masked_telea': 'global RGB mean fallback',
              'coalition_sampling': 'mandatory endpoints/singletons/complements; uniform unique remainder'}
    manifest_hash = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if args.stage == 'full':
        if not args.reviewed_pilot:
            parser.error('Full run requires --reviewed-pilot after visual inspection of the pilot')
        pilot = Path(args.reviewed_pilot)
        review = json.loads((pilot / 'review.json').read_text(encoding='utf-8'))
        report = json.loads((pilot / 'checks.json').read_text(encoding='utf-8'))
        previous = json.loads((pilot / 'config.json').read_text(encoding='utf-8'))
        if review.get('visual_checks_passed') is not True or report.get('all_passed') is not True:
            raise ValueError('Pilot automated checks and visual review must pass')
        if previous['experiment'] != config or previous['manifest_sha256'] != manifest_hash:
            raise ValueError('Full experiment settings or manifest differ from reviewed pilot')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; use a new --output directory')
    output.mkdir(parents=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # TF32 convolution can shift the original score by ~1e-4 between batch
    # sizes on Ada GPUs. Full-coalition validation uses full float32 precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = pairs[:3 if args.stage == 'pilot' else 30]
    save_json(output / 'config.json', {'experiment': config, 'manifest_sha256': manifest_hash,
        'stage': args.stage, 'device': device, 'arguments': vars(args)})
    save_json(output / 'selected_pairs.json', selected)
    save_json(output / 'environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None})
    scorer = ClipScorer(device, args.batch_size)
    inpainter = LamaInpainter(device, args.lama_weights)
    save_json(output / 'model_provenance.json', {'clip_commit': scorer.model.config._commit_hash,
        'lama_weights': str(inpainter.weights_path),
        'lama_sha256': hashlib.sha256(inpainter.weights_path.read_bytes()).hexdigest()})
    rows, reports, timings = [], [], []
    for i, pair in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {pair['image_id']}", flush=True)
        folder = output / f'image_{i:02d}'
        folder.mkdir()
        pil = ImageOps.exif_transpose(Image.open(pair['image_path'])).convert('RGB')
        pil.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
        image = np.array(pil)
        Image.fromarray(image).save(folder / 'original.png')
        segments = slic(image, n_segments=16, compactness=30, start_label=0, channel_axis=-1)
        _, segments = np.unique(segments, return_inverse=True)
        segments = segments.reshape(image.shape[:2])
        d = int(segments.max()) + 1
        coalitions = sample_coalitions(d)
        np.save(folder / 'segments.npy', segments)
        np.save(folder / 'coalitions.npy', coalitions)
        score = scorer.for_caption(pair['caption'])
        original_score = float(score([image])[0])
        values, curves, method_checks, endpoints = {}, {}, {}, {}
        examples = {}
        for method in METHODS:
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            predictions = []
            for offset in range(0, len(coalitions), args.batch_size):
                batch = []
                for index in range(offset, min(offset+args.batch_size, len(coalitions))):
                    mask = removal_mask(segments, coalitions[index], 3 if method == 'lama_expanded' else 0)
                    perturbed = mean_mask(image, mask) if method == 'mean' else inpainter(image, mask)
                    if perturbed.shape != image.shape or perturbed.dtype != np.uint8:
                        raise ValueError('Invalid perturbation shape/dtype')
                    batch.append(perturbed)
                predictions.extend(score(batch))
            phi, fit = kernel_shap(coalitions, predictions)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            timings.append({'image_id': pair['image_id'], 'method': method, 'explanation_seconds': elapsed,
                'peak_vram_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None})
            values[method] = phi
            endpoints[method] = float(predictions[0])
            np.save(folder / f'shap_{method}.npy', phi)
            np.save(folder / f'coalition_scores_{method}.npy', np.asarray(predictions))
            curves[method] = deletion_curve(image, segments, phi, score)
            method_checks[method] = {'full_score_matches_original': bool(np.isclose(predictions[1], original_score, atol=1e-6)),
                'nontrivial_shap': bool(np.ptp(phi) > 1e-8 and np.any(abs(phi) > 1e-8)),
                'efficiency_passed': bool(abs(fit['efficiency_residual']) < 1e-8), 'fit': fit,
                'ranking_descending_positive': bool(all(phi[j] > 0 for j in curves[method]['order']) and
                    all(phi[a] >= phi[b] for a,b in zip(curves[method]['order'], curves[method]['order'][1:])))}
        row = {'image_id': pair['image_id'], 'caption': pair['caption'],
               **{m + '_shap_deletion_auc': curves[m]['auc'] for m in METHODS}}
        rows.append(row)
        target_feature = int(np.argmax(values['mean']))
        example_idx = int(np.flatnonzero((coalitions.sum(1) == d - 1) & (coalitions[:, target_feature] == 0))[0])
        mask = removal_mask(segments, coalitions[example_idx])
        expanded = removal_mask(segments, coalitions[example_idx], 3)
        for method in METHODS:
            actual_mask = expanded if method == 'lama_expanded' else mask
            examples[method] = mean_mask(image, actual_mask) if method == 'mean' else inpainter(image, actual_mask)
            Image.fromarray(examples[method]).save(folder / f'perturbation_{method}.png')
        Image.fromarray(mask).save(folder / 'mask.png')
        Image.fromarray(expanded).save(folder / 'mask_expanded.png')
        report = {'image_id': pair['image_id'], 'n_segments': d, 'methods': method_checks,
            'original_score': original_score, 'example_coalition_index': example_idx,
            'example_removed_feature': target_feature,
            'lama_empty_endpoints_match': bool(np.isclose(endpoints['lama'], endpoints['lama_expanded'], rtol=0, atol=1e-6)),
            'shared_coalitions_sha256': hashlib.sha256(coalitions.tobytes()).hexdigest(),
            'mask_exact_absent_regions': bool(np.array_equal(mask > 0, coalitions[example_idx][segments] == 0)),
            'expansion_contains_original': bool(np.all(expanded[mask > 0] == 255)),
            'expansion_adds_pixels': bool(np.any((expanded > 0) & (mask == 0))),
            'source_image_sha256': hashlib.sha256(Path(pair['image_path']).read_bytes()).hexdigest()}
        reports.append(report)
        save_json(folder / 'checks.json', report)
        save_json(folder / 'deletion_curves.json', curves)
        save_json(folder / 'pair.json', pair)
        if i < 10:
            explanation_figure(folder / 'explanation.png', image, segments, values, curves, pair['caption'])
            perturbation_figure(folder / 'perturbations.png', image, mask, expanded, examples)
        with (output / 'results.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        save_json(output / 'timings.json', timings)
    passed = all(r['mask_exact_absent_regions'] and r['expansion_contains_original'] and r['expansion_adds_pixels'] and
        r['lama_empty_endpoints_match'] and
        all(all(c[k] for k in ['full_score_matches_original', 'nontrivial_shap', 'efficiency_passed',
                              'ranking_descending_positive']) for c in r['methods'].values()) for r in reports)
    save_json(output / 'checks.json', {'all_passed': passed, 'images': reports})
    summary = summarize(rows)
    summary['timing'] = {m: {'total_explanation_seconds': float(sum(t['explanation_seconds'] for t in timings if t['method'] == m)),
        'mean_explanation_seconds': float(np.mean([t['explanation_seconds'] for t in timings if t['method'] == m]))}
        for m in METHODS}
    save_json(output / 'summary.json', summary)
    if args.stage == 'pilot':
        save_json(output / 'review.json', {'visual_checks_passed': False,
            'instructions': 'Inspect all three perturbations.png and explanation.png figures. Verify intended feature removal, mask expansion, and plausible outputs. Set true only after review.'})
    print(f'Finished: {output.resolve()} (automated checks passed: {passed})', flush=True)


if __name__ == '__main__':
    main()
