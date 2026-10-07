"""Download and validate exactly the original Karpathy Flickr30k TEST images."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time
import threading
import zipfile
from urllib.parse import urlsplit

from PIL import Image
import requests

from experiment2.io import save_json, sha256
from .core import validate_annotations

ROOT = Path('data/flickr30k_karpathy_exp4')
ANNOTATION_URL = 'https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip'
MIRROR_URL = 'https://github.com/OpenGVLab/InternVL/releases/download/data/flickr30k_test_karpathy.json'
VIEWER_REVISION = '2b239befc81b6e3f035ce6bd52f5f4d60f5625f7'
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.


def get(url, **kwargs):
    global NEXT_REQUEST
    for attempt in range(6):
        try:
            if urlsplit(url).hostname == 'datasets-server.huggingface.co':
                with REQUEST_LOCK:
                    delay = max(0., NEXT_REQUEST-time.monotonic())
                    if delay:
                        time.sleep(delay)
                    NEXT_REQUEST = time.monotonic()+2.
            response = requests.get(url, timeout=(15, 30), **kwargs)
            if response.status_code == 429:
                cooldown = min(600., 120.*(attempt+1))
                with REQUEST_LOCK:
                    NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic()+cooldown)
                print(f'HTTP 429: pausing shared download requests for {cooldown:.0f}s; completed images preserved', flush=True)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            offset = kwargs.get('params', {}).get('offset', '')
            print(f'Download request retry {attempt+1}/6: {type(error).__name__}, status={status}, path={urlsplit(url).path}, offset={offset}', flush=True)
            if attempt == 5:
                raise
            time.sleep(1+attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--download-archive', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    annotation = ROOT/'dataset_flickr30k.json'
    if not annotation.exists():
        archive = get(ANNOTATION_URL).content
        (ROOT/'caption_datasets.zip').write_bytes(archive)
        annotation.write_bytes(zipfile.ZipFile(io.BytesIO(archive)).read('dataset_flickr30k.json'))
    document = json.loads(annotation.read_text(encoding='utf-8'))
    images, captions, validation = validate_annotations(document)
    mirror = ROOT/'flickr30k_test_karpathy.json'
    if not mirror.exists():
        mirror.write_bytes(get(MIRROR_URL).content)
    other = json.loads(mirror.read_text(encoding='utf-8'))
    mirror_images = {Path(i['file_name']).name.removesuffix('.jpg') for i in other['images']}
    # This COCO-format mirror retains CSV quoting for captions containing commas
    # and quotes. Decode that transport escaping only; use original raw text.
    def decode_mirror(text):
        return next(csv.reader([text]))[0] if text.startswith('"') and text.endswith('"') else text
    mirror_captions = {(str(a['image_id']), decode_mirror(a['caption'])) for a in other['annotations']}
    if (mirror_images != {r['image_id'] for r in images}
            or mirror_captions != {(r['image_id'], r['caption']) for r in captions}):
        raise ValueError('STOP: original Karpathy and retrieval mirror disagree')
    validation['independent_retrieval_mirror_matches'] = True
    validation['mirror_transport_decoding'] = 'CSV outer/doubled quotes decoded for cross-check only; original Karpathy captions used verbatim.'
    save_json(ROOT/'split_validation.json', validation)
    save_json(ROOT/'captions.json', captions)
    image_dir, sources = ROOT/'images', ROOT/'image_source_rows'
    image_dir.mkdir(exist_ok=True)
    sources.mkdir(exist_ok=True)
    by_id = {str(r['imgid']): r for r in document['images'] if r['split'] == 'test'}
    archive_provenance = None
    if args.download_archive:
        archive_path = ROOT/'flickr30k-images.zip'
        archive_url = f'https://huggingface.co/datasets/nlphuji/flickr30k/resolve/{VIEWER_REVISION}/flickr30k-images.zip?download=true'
        if not archive_path.exists():
            partial = archive_path.with_suffix('.zip.part')
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            print(f'Downloading original full image archive; resume offset {offset}', flush=True)
            with requests.get(archive_url, headers=headers, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                if offset and response.status_code == 206:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Archive resume range mismatch')
                    mode = 'ab'
                else:
                    offset, mode = 0, 'wb'
                total = offset+int(response.headers.get('Content-Length', 0))
                downloaded, last_logged = offset, offset
                started = time.monotonic()
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(4*1024*1024):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded-last_logged >= 128*1024*1024:
                            speed = (downloaded-offset)/max(time.monotonic()-started, .01)/1024**2
                            print(f'Archive: {downloaded/1024**3:.2f}/{total/1024**3:.2f} GiB ({speed:.1f} MiB/s)', flush=True)
                            last_logged = downloaded
                if total and downloaded != total:
                    raise ValueError('Incomplete image archive; partial retained for resume')
            partial.replace(archive_path)
        print('Archive downloaded; extracting exactly the 1000 original Karpathy test filenames', flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            members = {}
            expected = {r['filename'] for r in images}
            for member in archive.infolist():
                filename = Path(member.filename).name
                if filename in expected:
                    if filename in members:
                        raise ValueError(f'Ambiguous archive image: {filename}')
                    members[filename] = member
            if set(members) != expected:
                raise ValueError('STOP: archive missing standard test images')
            for i, record in enumerate(images):
                member = members[record['filename']]
                content = archive.read(member)
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
                path = image_dir/record['filename']
                path.write_bytes(content)
                save_json(sources/f"{record['image_id']}.json", {'source': 'pinned original image archive',
                    'archive_member': member.filename, 'sha256': sha256(path)})
                if (i+1) % 200 == 0:
                    print(f'Extracted and verified {i+1}/1000 test images', flush=True)
        archive_provenance = {'url': archive_url, 'sha256': sha256(archive_path), 'bytes': archive_path.stat().st_size}
        save_json(ROOT/'test_image_captions.json', [{'filename': r['filename'],
            'captions': [s['raw'] for s in by_id[str(r['karpathy_image_id'])]['sentences']]} for r in images])

    def download(record):
        path = image_dir/record['filename']
        evidence = sources/f"{record['image_id']}.json"
        if path.exists() and evidence.exists():
            previous = json.loads(evidence.read_text(encoding='utf-8'))
            if sha256(path) == previous['sha256']:
                return {**record, 'image_path': str(path.resolve()), 'sha256': previous['sha256']}
        params = {'dataset': 'nlphuji/flickr30k', 'config': 'TEST', 'split': 'test',
                  'offset': record['karpathy_image_id'], 'length': 1}
        response = get('https://datasets-server.huggingface.co/rows', params=params).json()
        item = response['rows'][0]['row']
        original = by_id[str(record['karpathy_image_id'])]
        if (item['filename'] != record['filename'] or item['split'] != 'test'
                or str(item['img_id']) != str(record['karpathy_image_id'])
                or list(map(int, item['sentids'])) != original['sentids']
                or item['caption'] != [s['raw'] for s in original['sentences']]
                or VIEWER_REVISION not in item['image']['src']):
            raise ValueError(f"STOP: image/annotation source mismatch for {record['image_id']}")
        content = get(item['image']['src']).content
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        temporary = path.with_suffix('.download')
        temporary.write_bytes(content)
        temporary.replace(path)
        digest = sha256(path)
        save_json(evidence, {'row_index': record['karpathy_image_id'], 'row': item, 'sha256': digest})
        return {**record, 'image_path': str(path.resolve()), 'sha256': digest}

    downloaded = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, row) for row in images]
        for future in as_completed(futures):
            row = future.result()
            downloaded[row['image_id']] = row
            if len(downloaded) % 50 == 0:
                print(f'Validated/downloaded {len(downloaded)}/1000 test images', flush=True)
    save_json(ROOT/'images.json', [downloaded[r['image_id']] for r in images])
    save_json(ROOT/'source.json', {'annotation_url': ANNOTATION_URL, 'annotation_sha256': sha256(annotation),
        'archive_sha256': sha256(ROOT/'caption_datasets.zip') if (ROOT/'caption_datasets.zip').exists() else None,
        'mirror_url': MIRROR_URL, 'mirror_sha256': sha256(mirror),
        'image_dataset': 'nlphuji/flickr30k', 'viewer_revision': VIEWER_REVISION,
        'image_archive': archive_provenance,
        'note': 'HF test is a mixed-partition container; only original Karpathy test rows are fetched and cross-validated.'})
    print('Standard Karpathy split and all 1000 test images validated', flush=True)


if __name__ == '__main__':
    main()
