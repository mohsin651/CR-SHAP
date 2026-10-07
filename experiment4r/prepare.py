"""Freeze census and validate original artifacts without writing to Experiment4."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from experiment2.io import save_json,sha256,verify_snapshot
from experiment2.model import load_image
from experiment4.run import CONFIG as CONFIG
from experiment4.screen import SCREEN,segment
from lili_shap.core import sample_coalitions
from .core import analyze,partition

ROOT=Path('results/experiment_4R_full_flickr30k')
INPUT=Path('data/experiment4r_preflight')
ORIGINAL=Path('results/experiment_4')


def source_hashes():
    return {str(p):sha256(p) for p in sorted(Path('experiment4r').glob('*.py'))}


def snapshot():
    paths=[]
    for folder in ('lili_shap','experiment2','experiment3','experiment4','experiment5','experiment6','experiment7'):
        paths.extend(p for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'pdf_tools' not in p.parts)
    paths.extend(p for p in Path('results').rglob('*') if p.is_file() and not any(part.startswith('experiment_4R') for part in p.parts))
    paths.extend(Path('.').glob('*.md')); paths.extend(Path('.').glob('requirements*.txt'))
    return {str(p):sha256(p) for p in sorted(paths)}


def main():
    if ROOT.exists() or INPUT.exists(): raise FileExistsError('4R output/preflight already exists')
    ROOT.mkdir(); INPUT.mkdir(parents=True); (INPUT/'preflight').mkdir()
    original_config=json.loads((ORIGINAL/'config.json').read_text())
    if original_config['experiment']!=CONFIG or original_config['gamma']!=100.: raise ValueError('Frozen Experiment4 config/gamma mismatch')
    selected=json.loads((ORIGINAL/'selected_images.json').read_text()); original_rows=json.loads((ORIGINAL/'rows.json').read_text())
    if [r['image_id'] for r in selected]!=[r['image_id'] for r in original_rows]: raise ValueError('Original300 manifest/rows mismatch')
    screening=json.loads((SCREEN/'rows.json').read_text())
    eligible,remainder,digest=partition(screening,[r['image_id'] for r in selected])
    if not json.loads((ORIGINAL/'verification.json').read_text())['all_passed']: raise ValueError('Original verified results required')
    if analyze(original_rows)!=json.loads((ORIGINAL/'summary.json').read_text()): raise ValueError('Original300 aggregate statistics do not reproduce exactly')
    save_json(ROOT/'population_manifest.json',eligible); save_json(ROOT/'original300_manifest.json',selected); save_json(ROOT/'new493_manifest.json',remainder)
    save_json(ROOT/'scope.json',{'original_eligible_intended':793,'original_reused':300,'new_intended':493,'eligible_id_set_sha256':digest,
        'hash_encoding':'SHA256 of UTF8 JSON of lexicographically sorted image IDs, separators comma/colon',
        'full_analysis_order':'Original300 in unchanged original order, followed by new493 in lexicographic image-ID order',
        'design':'Post-specified finite eligible-test-population robustness; not the original sampled Experiment4',
        'interpretation_frozen_before_new_outcomes':'Report exact full versus original effect difference. Similar positive magnitudes support descriptive representativeness; smaller positive effects mean original overestimation; uncertain/nonpositive effects weaken or reverse original interpretation. No thresholds tuned to outcomes and no equivalence claims.'})
    save_json(ROOT/'original_protocol_config.json',original_config)
    protected=snapshot(); save_json(ROOT/'protected_experiments.json',protected); save_json(INPUT/'protected_experiments.json',protected)
    registry=[]
    for i,row in enumerate(original_rows):
        folder=ORIGINAL/f'example_{i:03d}'
        registry.append({'image_id':row['image_id'],'artifact_directory':str(folder),'original_index':i,
            'files':{p.name:sha256(p) for p in folder.iterdir() if p.is_file()}})
    save_json(ROOT/'reused_artifacts.json',registry)
    # Execute the original read-only checks for the first five examples, with
    # its sole output redirected to this new directory. Original files untouched.
    import experiment4.verify as original_verify
    code=inspect.getsource(original_verify.verify)
    code=code.replace("rows = json.loads((root/'rows.json').read_text())","rows = json.loads((root/'rows.json').read_text())[:5]")
    code=code.replace("expected_n = 5 if config['stage'] == 'pilot' else 300","expected_n = 5")
    code=code.replace("require(analyze(rows) == json.loads((root/'summary.json').read_text()), 'recomputed paired inference')","pass  # Full original300 aggregate reproduction checked separately above.")
    (ROOT/'original_subset_verifier.py').write_text(code,encoding='utf-8')
    namespace=dict(vars(original_verify))
    def redirect(path,value):
        if Path(path)!=ORIGINAL/'verification.json': raise ValueError('Unexpected original-output write')
        save_json(ROOT/'reused_subset_verification.json',value)
    namespace['save_json']=redirect
    exec(compile(code,'original_subset_verifier.py','exec'),namespace)
    namespace['verify'](ORIGINAL)
    checks,failures,feasible=[],[],[]
    for index,row in enumerate(remainder):
        try:
            if sha256(row['image_path'])!=row['sha256']: raise ValueError('Original image hash mismatch')
            image=load_image(row['image_path']); labels=segment(image); d=int(labels.max())+1
            path=INPUT/'preflight'/f"{row['image_id']}.npy"; np.save(path,labels)
            sample_coalitions(d,128,42)
            checks.append({'image_id':row['image_id'],'num_superpixels':d,'maximum_unique_coalitions':2**d,'feasible':True,
                'segmentation_sha256':sha256(path),'working_image_sha256':hashlib.sha256(image.tobytes()).hexdigest()})
            feasible.append(row)
        except (ValueError,OSError) as error:
            failures.append({'image_id':row['image_id'],'new_population_index':index,'phase':'preflight','reason':str(error),'no_replacement':True})
    save_json(ROOT/'technical_failures_preflight.json',failures)
    save_json(INPUT/'selected_images.json',feasible); save_json(INPUT/'segmentation_preflight.json',{'all_passed':True,'checks':checks,
        'new_intended':493,'feasible':len(feasible),'failure_count':len(failures),'all_checked_before_new_explanation_outcomes':True})
    save_json(INPUT/'technical_exclusions.json',failures)
    save_json(INPUT/'selection_report.json',{'design':'Full frozen eligible remainder; no sampling or replacement','new_intended':493,'feasible':len(feasible),'original_reused':300,'full_intended':793})
    save_json(INPUT/'screening_summary.json',json.loads((SCREEN/'screening_summary.json').read_text()))
    save_json(INPUT/'screening_artifact_hashes.json',json.loads((SCREEN/'screening_artifact_hashes.json').read_text()))
    save_json(INPUT/'preflight_hashes.json',{str(p.relative_to(INPUT)):sha256(p) for p in INPUT.rglob('*') if p.is_file()})
    save_json(ROOT/'pre_run_checks.json',{'original793_exact':True,'original300_subset':True,'remainder493_disjoint':True,
        'original_summary_exactly_reproduced':True,'original_artifacts_hashed':300,'original_examples_independently_reconstructed':5,
        'new_preflight_complete':True,'new_preflight_feasible':len(feasible),'failures':len(failures),'source_hashes':source_hashes()})
    verify_snapshot(protected); print(f'4R integrity verified: original300 reused exactly; {len(feasible)}/493 new feasible, {len(failures)} failures; no sampling or new retrieval',flush=True)


if __name__=='__main__': main()
