import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import requests


def prepare(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'source.json').exists():
        raise FileExistsError('Dataset already prepared; reuse it rather than downloading again')
    session = requests.Session()
    session.headers['User-Agent'] = 'clip-cr-shap-prototype'
    response = session.get('https://api.github.com/repos/RAIVNLab/sugar-crepe/commits/main', timeout=30)
    response.raise_for_status()
    revision = response.json()['sha']
    response = session.get(f'https://api.github.com/repos/RAIVNLab/sugar-crepe/contents/data?ref={revision}', timeout=30)
    response.raise_for_status()
    annotations = root / 'annotations'
    annotations.mkdir(exist_ok=True)
    sources = []
    for item in response.json():
        if not item['name'].endswith('.json'):
            continue
        url = f"https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/{revision}/data/{item['name']}"
        response = session.get(url, timeout=30)
        response.raise_for_status()
        path = annotations / item['name']
        if path.exists():
            if path.read_bytes() != response.content:
                raise ValueError('Existing annotations do not match pinned revision')
        else:
            path.write_bytes(response.content)
        sources.append({'category': path.stem, 'url': url, 'sha256': hashlib.sha256(response.content).hexdigest()})
    # Path-style HTTPS reaches the official COCO S3 bucket with a valid certificate.
    url = 'https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip'
    archive = root / 'val2017.zip'
    if not archive.exists():
        partial = root / 'val2017.zip.partial'
        if partial.exists():
            raise FileExistsError(f'Incomplete download exists: {partial}; use a new data directory')
        with session.get(url, stream=True, timeout=(30, 60)) as response:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            downloaded = 0
            progress = 0
            with partial.open('xb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - progress >= 50 * 1024 * 1024:
                        print(f'COCO download: {downloaded/1e6:.0f}/{total/1e6:.0f} MB', flush=True)
                        progress = downloaded
            if total and downloaded != total:
                raise ValueError('Incomplete COCO archive')
        partial.rename(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError('Archive path escapes dataset directory')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise ValueError(f'Incomplete existing extracted image: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open('xb') as dest:
                shutil.copyfileobj(source, dest)
    (root / 'source.json').write_text(json.dumps({'repository': 'RAIVNLab/sugar-crepe',
        'revision': revision, 'annotations': sources, 'images_url': url,
        'images_archive_sha256': digest.hexdigest()}, indent=2), encoding='utf-8')
    print(f'Dataset prepared: {root.resolve()}', flush=True)


def load_instances(root):
    root = Path(root).resolve()
    instances = []
    for path in sorted((root / 'annotations').glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for source_id, item in data.items():
            image_path = root / 'val2017' / item['filename']
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            instances.append({'example_id': f'{path.stem}:{source_id}',
                'source_id': str(source_id), 'image_id': item['filename'],
                'image_path': str(image_path), 'sugarcrepe_category': path.stem,
                'positive_caption': item['caption'], 'negative_caption': item['negative_caption']})
    if not instances or len({r['example_id'] for r in instances}) != len(instances):
        raise ValueError('Missing or duplicate benchmark examples')
    return instances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', default='data/sugarcrepe')
    prepare(parser.parse_args().destination)
