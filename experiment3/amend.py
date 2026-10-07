"""Apply the user-authorized single technical exclusion, preserving provenance."""
import json
import shutil
from collections import Counter
from pathlib import Path

from experiment2.io import save_json, sha256, verify_snapshot


def main():
    workspace = Path.cwd().resolve()
    screening = workspace/'data/sugarcrepe_screening_exp3'
    archive = workspace/'results/experiment_3_stopped_001'
    pilot_archive = workspace/'results/experiment_3_pilot_original'
    if archive.exists() or pilot_archive.exists(): raise FileExistsError('Amendment already applied')
    verify_snapshot(json.loads((screening/'protected_experiments.json').read_text()))
    preflight = json.loads((screening/'segmentation_preflight.json').read_text())
    if preflight['failures'] != [{'example_id':'swap_att:39','num_superpixels':6,'supports_128_unique_coalitions':False}]:
        raise ValueError('Unexpected technical failure set')
    for name in ['confirmatory_selected_instances.json','selection_report.json','segmentation_preflight.json']:
        shutil.copy2(screening/name,screening/(name+'.original_700'))
    for old,new in [(workspace/'results/experiment_3',archive),(workspace/'results/experiment_3_pilot',pilot_archive)]:
        if not old.resolve().is_relative_to(workspace/'results') or not new.resolve().is_relative_to(workspace/'results'):
            raise ValueError('Archive path outside intended results directory')
        old.rename(new)
    original = json.loads((screening/'confirmatory_selected_instances.json').read_text())
    selected = [r for r in original if r['example_id']!='swap_att:39']
    assert len(original)==700 and len(selected)==699
    amendment = {'authorized_by':'User requested fixing and running after the proposed one-instance technical exclusion',
        'excluded_example_id':'swap_att:39','reason':'Six fixed SLIC regions permit only 64 distinct coalitions; frozen sampler requires 128 unique coalitions',
        'original_sample_size':700,'amended_sample_size':699,'replacement':False,'method_settings_changed':False,
        'original_selection_sha256':sha256(screening/'confirmatory_selected_instances.json.original_700'),
        'disclosure':'Amendment after 55 ordered instances had been evaluated; exclusion based only on coalition feasibility, not attribution outcomes.'}
    save_json(screening/'protocol_amendment.json',amendment)
    save_json(screening/'confirmatory_selected_instances.json',selected)
    report = json.loads((screening/'selection_report.json').read_text())
    counts = Counter(r['image_id'] for r in selected)
    report.update({'n':699,'unique_images':len(counts),'images_repeated':sum(n>1 for n in counts.values()),
        'maximum_instances_per_image':max(counts.values()),'protocol_amendment':amendment})
    report['categories']['swap_att'].update({'selected':99,'selected_unique_images':99,'shortage':1,'shortage_reason':'Authorized technical exclusion, not a shortage of eligible benchmark instances'})
    save_json(screening/'selection_report.json',report)
    preflight['examples'] = [r for r in preflight['examples'] if r['example_id']!='swap_att:39']
    preflight.update({'all_passed':True,'n':699,'failures':[],'excluded_technical_case':amendment})
    save_json(screening/'segmentation_preflight.json',preflight)
    print('Archived original attempts; amended sample: 699; frozen method unchanged')


if __name__=='__main__': main()
