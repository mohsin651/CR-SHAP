import json
from pathlib import Path

import numpy as np

from experiment2.io import save_json,sha256,verify_snapshot
from .core import analyze,grounding,union_mask
from .prepare import CONFIG,EXP4,ROOT,source_hashes


def main():
    config=json.loads((ROOT/'config.json').read_text())
    if config['experiment']!=CONFIG or config['source_hashes']!=source_hashes(): raise ValueError('Configuration/source changed')
    rows=json.loads((ROOT/'rows.json').read_text()); matched=json.loads((ROOT/'matched_manifest.json').read_text())
    excluded=json.loads((ROOT/'exclusions.json').read_text())
    if len(rows)+len(excluded)!=300 or [r['image_id'] for r in rows]!=[r['image_id'] for r in matched]: raise ValueError('Sample mismatch')
    for row,record in zip(rows,matched):
        i=row['experiment4_index']; folder=ROOT/'examples'/f'{i:03d}'
        source=json.loads((folder/'source.json').read_text())
        if source['raw_sha256']!=sha256(EXP4/f'example_{i:03d}'/'raw.npz'): raise ValueError('Source attribution changed')
        with np.load(folder/'grounding.npz') as raw, np.load(EXP4/f'example_{i:03d}'/'raw.npz') as old:
            if not np.array_equal(raw['segments'],old['segments']): raise ValueError('SLIC mismatch')
            target=union_mask(record['transformed_boxes'],raw['segments'].shape)
            if not np.array_equal(target,raw['gt_union']): raise ValueError('GT raster mismatch')
            for prefix,key in [('point','phi_pointwise'),('cr','phi_cr_shap')]:
                if not np.array_equal(raw[prefix+'_phi'],old[key]): raise ValueError('SHAP changed')
                result,pixels=grounding(old[key],old['segments'],target)
                if not np.array_equal(pixels,raw[prefix+'_positive_density']): raise ValueError('Density reconstruction mismatch')
                if any(row[prefix+'_'+k]!=v for k,v in result.items()): raise ValueError('Metric mismatch')
                # Independent energy calculation directly from region box overlap.
                counts=np.bincount(old['segments'].ravel(),minlength=len(old[key]))
                inside=np.bincount(old['segments'][target],minlength=len(old[key]))
                expected=float((np.maximum(old[key],0)*inside/counts).sum())
                if not np.isclose(expected,row[prefix+'_eib_numerator'],atol=1e-12): raise ValueError('Energy formula mismatch')
                if row[prefix+'_point_xy'] is not None:
                    x,y=row[prefix+'_point_xy']
                    hit=any(x+.5>=a and x+.5<c and y+.5>=b and y+.5<d for a,b,c,d in record['transformed_boxes'])
                    if hit!=row[prefix+'_pointing_game']: raise ValueError('Continuous-box point mismatch')
    if analyze(rows)!=json.loads((ROOT/'summary.json').read_text()): raise ValueError('Paired statistics changed')
    for relative,digest in json.loads((ROOT/'target_hashes.json').read_text()).items():
        if sha256(ROOT/relative)!=digest: raise ValueError('Frozen annotation target changed')
    protected=json.loads((ROOT/'protected_experiments.json').read_text()); verify_snapshot(protected)
    save_json(ROOT/'verification.json',{'all_passed':True,'n':len(rows),'protected_files_unchanged':len(protected),
        'scope':'Unchanged reused SHAP/SLIC, GT union raster, mass-conserving density, continuous-box pointing, independent region-overlap energy, area and paired statistics; no model inference.'})
    print('Experiment6 saved evidence verified; previous experiments unchanged',flush=True)


if __name__=='__main__': main()
