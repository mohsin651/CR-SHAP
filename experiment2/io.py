import csv
import hashlib
import json
from pathlib import Path


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def experiment1_snapshot():
    paths = list(Path('lili_shap').glob('*.py')) + [Path(p) for p in
        ['project_proposal.md', 'experiment_report.md', 'README.md', 'requirements.txt', 'requirements-lock.txt']]
    paths += [p for p in Path('results').rglob('*') if p.is_file() and
              not any(part.startswith('experiment_2') for part in p.parts)]
    return {str(p): sha256(p) for p in sorted(paths)}


def verify_snapshot(snapshot):
    changed = [p for p, digest in snapshot.items() if not Path(p).exists() or sha256(p) != digest]
    if changed:
        raise ValueError(f'Experiment 1 files changed: {changed}')
