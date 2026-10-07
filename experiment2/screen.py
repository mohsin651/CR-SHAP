import argparse
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from .core import stratified_select
from .data import load_instances
from .io import experiment1_snapshot, save_json, verify_snapshot, write_csv
from .model import PairClipScorer, load_image


def configure():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/sugarcrepe')
    parser.add_argument('--output', default='data/sugarcrepe_screening')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    instances = load_instances(args.data)
    snapshot = experiment1_snapshot()
    output.mkdir(parents=True)
    save_json(output / 'experiment1_integrity.json', snapshot)
    configure()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = PairClipScorer(device)
    captions = list(dict.fromkeys(c for r in instances for c in [r['positive_caption'], r['negative_caption']]))
    image_paths = list(dict.fromkeys(r['image_path'] for r in instances))
    started = time.perf_counter()
    texts = scorer.encode_texts(captions)
    np.save(output / 'text_embeddings.npy', texts)
    save_json(output / 'captions.json', captions)
    features = []
    for offset in range(0, len(image_paths), 16):
        batch = [load_image(path) for path in image_paths[offset:offset+16]]
        features.append(scorer.encode_images(batch))
        if offset % 160 == 0:
            print(f'Screening images: {min(offset+16,len(image_paths))}/{len(image_paths)}', flush=True)
    features = np.concatenate(features)
    np.save(output / 'image_embeddings.npy', features)
    save_json(output / 'image_paths.json', image_paths)
    caption_index = {c: i for i,c in enumerate(captions)}
    image_index = {p: i for i,p in enumerate(image_paths)}
    gamma = scorer.gamma
    for row in instances:
        feature = features[image_index[row['image_path']]]
        # Same numpy dot convention as pair scoring, sharing one image embedding.
        pair = texts[[caption_index[row['positive_caption']], caption_index[row['negative_caption']]]]
        scores = feature @ pair.T
        margin = float(scores[0]) - float(scores[1])
        row.update({'original_positive_cosine': float(scores[0]), 'original_negative_cosine': float(scores[1]),
            'original_margin': margin, 'original_preference': float(expit(gamma * margin)),
            'correctly_ranked': margin > 0})
    write_csv(output / 'screening.csv', instances)
    def accuracy(rows):
        correct = sum(r['correctly_ranked'] for r in rows)
        return {'n_examined': len(rows), 'n_correct': correct, 'n_incorrect': len(rows)-correct,
                'n_ties': sum(r['original_margin'] == 0 for r in rows), 'accuracy': correct/len(rows)}
    selected, allocation = stratified_select(instances)
    save_json(output / 'selected_instances.json', selected)
    save_json(output / 'screening_summary.json', {'overall': accuracy(instances),
        'categories': {c: accuracy([r for r in instances if r['sugarcrepe_category'] == c])
                       for c in sorted(allocation)}, 'allocation': allocation, 'gamma': gamma,
        'seed': 42, 'selection_rule': 'Seeded balanced water filling across official categories, then uniform sampling without replacement among correctly ranked instances.',
        'unique_images': len(image_paths), 'unique_captions': len(captions),
        'screening_seconds': time.perf_counter()-started,
        'clip_commit': scorer.model.config._commit_hash, 'device': device,
        'preprocessing': 'Experiment 1: EXIF transpose, RGB, aspect-preserving max side 512, then identical CLIPProcessor.',
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}})
    verify_snapshot(snapshot)
    print(f'Screened {len(instances)} instances; selected 100: {allocation}', flush=True)


if __name__ == '__main__':
    main()
