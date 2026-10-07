"""Independent saved-evidence checks without repeating model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit

from experiment2.io import save_json, verify_snapshot
from lili_shap.core import mean_mask, positive_order, removal_mask
from .core import BUDGETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',nargs='?',default='results/experiment_3')
    args = parser.parse_args()
    root = Path(args.directory)
    rows = json.loads((root/'rows.json').read_text())
    selected = json.loads((root/'selected_instances.json').read_text())
    assert [r['example_id'] for r in rows]==[r['example_id'] for r in selected[:len(rows)]]
    assert len({r['example_id'] for r in rows})==len(rows)
    previous = json.loads(Path('results/experiment_2/selected_instances.json').read_text())
    assert not {r['example_id'] for r in rows}&{r['example_id'] for r in previous}
    max_error = 0.
    for index,row in enumerate(rows):
        folder = root/f'example_{index:03d}'
        raw = np.load(folder/'raw.npz')
        image = np.asarray(Image.open(folder/'original.png'))
        segments,coalitions = raw['segments'],raw['coalitions']
        d = row['num_superpixels']
        assert coalitions.shape==(128,d) and len(np.unique(coalitions,axis=0))==128
        assert np.array_equal(coalitions[0],np.zeros(d)) and np.array_equal(coalitions[1],np.ones(d))
        for j in range(d):
            singleton = np.eye(d,dtype=np.uint8)[j]
            assert np.any(np.all(coalitions==singleton,axis=1))
            assert np.any(np.all(coalitions==1-singleton,axis=1))
        for i,coalition in enumerate(coalitions):
            mask = removal_mask(segments,coalition)
            assert hashlib.sha256(mask.tobytes()).hexdigest()==raw['mask_sha256'][i]
            assert hashlib.sha256(mean_mask(image,mask).tobytes()).hexdigest()==raw['perturbation_sha256'][i]
        error = float(np.max(abs(raw['phi_cr_shap']-(raw['phi_pointwise']-raw['phi_negative']))))
        assert error<1e-10
        max_error = max(max_error,error)
        positive = json.loads((folder/'positive_deletion.json').read_text())
        matched = json.loads((folder/'matched_deletion.json').read_text())
        for prefix,phi_key in [('point','phi_pointwise'),('cr','phi_cr_shap')]:
            phi = raw[phi_key]
            curve = positive[prefix]
            assert curve['order']==positive_order(phi).tolist()
            mask = np.zeros(segments.shape,bool)
            fractions = [0.]
            for j in curve['order']:
                mask |= segments==j
                fractions.append(float(mask.mean()))
            assert np.allclose(fractions,curve['fractions'],atol=0,rtol=0)
            assert np.allclose(expit(100*np.asarray(curve['margins'])),curve['preferences'],atol=1e-12,rtol=0)
            assert np.isclose(np.trapz(curve['auc_scores'],curve['auc_fractions']),row[prefix+'_positive_pdauc'],atol=1e-12,rtol=0)
            full = matched[prefix]
            assert full['order']==sorted(range(d),key=lambda j:(-phi[j],j))
            assert sorted(full['order'])==list(range(d))
            masks = raw[prefix+'_matched_masks']
            assert masks.shape==(d+1,*segments.shape)
            mask[:] = False
            for state in range(d+1):
                if state: mask |= segments==full['order'][state-1]
                assert np.array_equal(masks[state]>0,mask)
                assert full['fractions'][state]==float(mask.mean())
            assert np.allclose(np.asarray(full['positive_cosines'])-np.asarray(full['negative_cosines']),full['margins'],atol=1e-12,rtol=0)
            assert np.allclose(expit(100*np.asarray(full['margins'])),full['preferences'],atol=1e-12,rtol=0)
            for q in BUDGETS:
                state = full['budget_states'][str(q)]
                assert full['fractions'][state-1]<q/100<=full['fractions'][state]
                assert row[f'{prefix}_actual_area{q}']==full['fractions'][state]
                assert row[f'{prefix}_preference{q}']==full['preferences'][state]
                assert row[f'{prefix}_rankflip{q}']==(full['margins'][state]<0)
                assert np.isclose(row[f'{prefix}_margin_drop{q}'],row['original_margin']-full['margins'][state],atol=1e-12,rtol=0)
            assert full['auc_fractions'][-1]==row[prefix+'_actual_area50']
            assert np.isclose(np.trapz(full['auc_preferences'],full['auc_fractions'])/full['auc_fractions'][-1],row[prefix+'_matched_pdauc50'],atol=1e-12,rtol=0)
        if (index+1)%100==0: print(f'Validated {index+1}/{len(rows)} saved examples',flush=True)
    protected = json.loads((root/'protected_experiments.json').read_text())
    verify_snapshot(protected)
    save_json(root/'verification.json',{'all_passed':True,'n':len(rows),'shared_coalitions_validated':128*len(rows),
        'max_abs_linearity_error':max_error,'protected_files_unchanged':len(protected),
        'scope':'Coalition hashes, endpoints, singleton/complements, Shapley linearity, positive/signed rankings, saved cumulative masks, first budget crossings, preference and AUC formulas; no new model inference.'})
    print('Saved evidence validation passed',flush=True)


if __name__=='__main__':
    main()
