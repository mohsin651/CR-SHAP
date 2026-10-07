import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import requests


def download_flickr30k(destination, count=30, seed=42):
    """Download only seeded sampled images through the HF dataset viewer."""
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / 'pairs.csv'
    if manifest.exists():
        raise FileExistsError(f'Refusing to overwrite {manifest}')
    session = requests.Session()
    params = {'dataset': 'nlphuji/flickr30k', 'config': 'TEST', 'split': 'test'}
    def rows(offset, length=1):
        response = session.get('https://datasets-server.huggingface.co/rows',
                               params={**params, 'offset': int(offset), 'length': length}, timeout=60)
        response.raise_for_status()
        return response.json()
    first = rows(0)
    total = first['num_rows_total']
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=count, replace=False)
    records, provenance = [], []
    for i, index in enumerate(indices):
        item = rows(index)['rows'][0]['row']
        captions = item['caption']
        caption = captions[int(rng.integers(len(captions)))] if isinstance(captions, list) else captions
        image_id = str(item.get('img_id', item.get('filename', index)))
        filename = f'{int(index):05d}.jpg'
        url = item['image']['src']
        response = session.get(url, timeout=60)
        response.raise_for_status()
        target = dest / filename
        if target.exists():
            raise FileExistsError(target)
        target.write_bytes(response.content)
        records.append({'image_id': image_id, 'image_path': filename, 'caption': caption})
        provenance.append({'row_index': int(index), 'image_id': image_id, 'source_row': item,
                           'sha256': hashlib.sha256(response.content).hexdigest()})
        print(f'Downloaded Flickr30k {i+1}/{count}: {image_id}', flush=True)
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_id', 'image_path', 'caption'])
        writer.writeheader()
        writer.writerows(records)
    (dest / 'source.json').write_text(json.dumps({'dataset': params, 'seed': seed,
        'sampling_population': total, 'records': provenance}, indent=2), encoding='utf-8')
    return manifest


def load_pairs(manifest):
    path = Path(manifest).resolve()
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {'image_id', 'image_path', 'caption'}.issubset(rows[0]):
        raise ValueError('Manifest needs image_id,image_path,caption columns')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest must have exactly one caption per unique image')
    for row in rows:
        image = Path(row['image_path'])
        row['image_path'] = str(image if image.is_absolute() else path.parent / image)
        if not Path(row['image_path']).exists() or not row['caption'].strip():
            raise ValueError(f'Missing image or caption: {row}')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', default='data/flickr30k')
    parser.add_argument('--count', type=int, default=30)
    args = parser.parse_args()
    download_flickr30k(args.destination, args.count)
