"""Gate screening, protect existing experiments, freeze the new sample."""
import csv
import json
from pathlib import Path

from experiment2.io import save_json, sha256
from .core import select_sample


def main():
    screening = Path('data/sugarcrepe_screening_exp3')
    summary = json.loads((screening/'screening_summary.json').read_text())
    expected = {'n_examined':7511,'n_correct':5745,'n_incorrect':1766,'n_ties':0}
    if any(summary['overall'][k] != v for k,v in expected.items()):
        raise ValueError('STOP: reproduced screening counts differ')
    if summary['clip_commit'] != '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268' or summary['gamma'] != 100.:
        raise ValueError('STOP: screening model differs')
    with (screening/'screening.csv').open(encoding='utf-8',newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for k in ['original_positive_cosine','original_negative_cosine','original_margin','original_preference']:
            row[k] = float(row[k])
        row['correctly_ranked'] = row['original_margin'] > 0
    previous = json.loads(Path('results/experiment_2/selected_instances.json').read_text())
    selected, report = select_sample(rows,{r['example_id'] for r in previous})
    target = screening/'confirmatory_selected_instances.json'
    if target.exists():
        raise FileExistsError(target)
    save_json(target,selected)
    save_json(screening/'selection_report.json',report)
    protected = [p for root in ['results','lili_shap','experiment2'] for p in Path(root).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts]
    protected += [Path(p) for p in ['README.md','experiment_report.md','project_proposal.md','requirements.txt','requirements-lock.txt']]
    save_json(screening/'protected_experiments.json',{str(p):sha256(p) for p in sorted(protected)})
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
