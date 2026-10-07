"""Verify saved gradient maps, ownership, reused SHAP and evaluator evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.special import expit
import torch

from experiment2.io import save_json, sha256, verify_snapshot
from experiment2.model import load_image
from experiment3.core import BUDGETS
from lili_shap.core import positive_order
from .core import analyze
from .model import map_from_gradient, project_map, region_means, spatial_weights
from .run import CONFIG, EXP4, source_hashes


def require(condition, message):
    if not condition: raise ValueError('Saved-evidence validation failed: '+message)


def verify(root):
    config = json.loads((root/'config.json').read_text())
    require(config['experiment']==CONFIG and config['source_hashes']==source_hashes(),'method/source hashes')
    require(config['selection_sha256']==sha256(EXP4/'selected_images.json'),'frozen sample hash')
    require(config['shap_results_sha256']==sha256(EXP4/'rows.json'),'frozen SHAP results hash')
    rows = json.loads((root/'rows.json').read_text())
    selected = json.loads((root/'selected_images.json').read_text())
    shap_rows = json.loads((EXP4/'rows.json').read_text())
    n = 5 if config['stage']=='pilot' else 300
    require(len(rows)==n and [r['image_id'] for r in rows]==[r['image_id'] for r in selected[:n]],'exact sample prefix')
    integrity = json.loads((root/'model_integrity.json').read_text())
    require(integrity['unchanged'] and integrity['before_sha256']==integrity['after_sha256'],'unchanged parameters')
    gamma = integrity['gamma']
    # Loading the processor alone does not instantiate CLIP or rerun inference.
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained(CONFIG['clip_model'],revision=CONFIG['clip_commit'],local_files_only=True)
    zeros = {'grad_point':[], 'grad_cr':[]}
    for i,row in enumerate(rows):
        folder = root/f'example_{i:03d}'
        require(json.loads((folder/'instance.json').read_text())==selected[i],'instance/caption ownership')
        for prefix,method in (('point','shap_point'),('cr','shap_cr')):
            for key,value in shap_rows[i].items():
                if key.startswith(prefix+'_'):
                    require(row[method+key[len(prefix):]]==value,'verbatim SHAP metrics')
        for key in ('positive_caption_id','negative_caption_id','positive_caption','negative_caption','negative_caption_source_image_id',
                    'original_positive_cosine','original_negative_cosine','original_margin','original_preference'):
            require(row[key]==shap_rows[i][key],'original image/caption scores')
        validation = json.loads((folder/'validation.json').read_text())
        require(validation['native_score_max_abs_error']<1e-6 and validation['gradient_linearity_error']<1e-6
                and validation['repeat_gradient_error']==0,'gradient/native-forward technical checks')
        image = np.asarray(Image.open(EXP4/f'example_{i:03d}'/'original.png'))
        with np.load(folder/'raw.npz') as raw:
            segments = raw['segments']
            with np.load(EXP4/f'example_{i:03d}'/'raw.npz') as old:
                require(np.array_equal(segments,old['segments']) and np.array_equal(raw['text_embeddings'],old['text_embeddings']),'saved SLIC and texts')
            weights = spatial_weights(torch.from_numpy(raw['queries']),torch.from_numpy(raw['keys'])).numpy()
            require(np.allclose(weights,raw['spatial_weights'],atol=2e-6,rtol=0),'official spatial weight formula')
            require(np.allclose(raw['grad_cr_gradient_cls'],raw['grad_point_gradient_cls']-raw['negative_gradient_cls'],atol=1e-6,rtol=0),'contrastive gradient linearity')
            matched = json.loads((folder/'matched_deletion.json').read_text())
            positive = json.loads((folder/'positive_deletion.json').read_text())
            for method in ('grad_point','grad_cr'):
                signed,relevance = map_from_gradient(torch.from_numpy(raw[method+'_gradient_cls']),torch.from_numpy(raw['values']),torch.from_numpy(raw['spatial_weights']))
                require(np.allclose(signed.numpy().reshape(7,7),raw[method+'_signed_patch_map'],atol=1e-6,rtol=0),'pre-ReLU map')
                require(np.allclose(relevance.numpy().reshape(7,7),raw[method+'_patch_map'],atol=1e-6,rtol=0),'native ReLU map')
                pixels,geometry = project_map(torch.from_numpy(raw[method+'_patch_map']),image,processor)
                require(np.allclose(pixels,raw[method+'_pixel_map'],atol=1e-6,rtol=0),'crop/resize inverse projection')
                require(geometry==json.loads((folder/'map_geometry.json').read_text()),'map geometry')
                regions = region_means(raw[method+'_pixel_map'],segments)
                require(np.array_equal(regions,raw[method+'_region_scores']),'mean SLIC relevance')
                require(np.isfinite(pixels).all() and np.any(raw[method+'_gradient_cls']!=0),'finite maps/nonzero gradients')
                if not np.any(raw[method+'_patch_map']>0): zeros[method].append(row['image_id'])
                curve = matched[method]
                order = sorted(range(len(regions)),key=lambda j:(-regions[j],j))
                require(curve['order']==order,'descending signed/all-region rank')
                mask = np.zeros(segments.shape,dtype=bool)
                for state in range(len(regions)+1):
                    if state: mask |= segments==order[state-1]
                    require(np.array_equal(raw[method+'_matched_masks'][state]>0,mask),'whole-region cumulative mask')
                    require(curve['fractions'][state]==float(mask.mean()),'actual pixel area')
                margin = np.asarray(curve['positive_cosines'])-np.asarray(curve['negative_cosines'])
                require(np.allclose(margin,curve['margins'],atol=1e-12,rtol=0),'cosine margins')
                require(np.allclose(expit(gamma*margin),curve['preferences'],atol=1e-12,rtol=0),'gamma preference')
                states = [0]
                for q in BUDGETS:
                    s = curve['budget_states'][str(q)]; states.append(s)
                    require(curve['fractions'][s-1]<q/100<=curve['fractions'][s],'first budget crossing')
                    require(row[f'{method}_actual_area{q}']==curve['fractions'][s] and row[f'{method}_preference{q}']==curve['preferences'][s],'budget metrics')
                    require(row[f'{method}_rankflip{q}']==(margin[s]<0),'rank flip')
                    require(np.isclose(row[f'{method}_margin_drop{q}'],row['original_margin']-margin[s],atol=1e-12),'margin drop')
                x,y = [curve['fractions'][s] for s in states],[curve['preferences'][s] for s in states]
                require(x==curve['auc_fractions'] and y==curve['auc_preferences'],'six-state primary AUC')
                require(np.isclose(np.trapz(y,x)/x[-1],row[method+'_matched_pdauc50'],atol=1e-12),'matched PDAUC')
                pos = positive[method]
                require(pos['order']==positive_order(regions).tolist(),'positive-only native support')
                mask[:] = False
                fractions = [0.]
                for j in pos['order']:
                    mask |= segments==j; fractions.append(float(mask.mean()))
                require(fractions==pos['fractions'],'positive region fractions')
                pm = np.asarray(pos['positive_cosines'])-np.asarray(pos['negative_cosines'])
                require(np.allclose(expit(gamma*pm),pos['preferences'],atol=1e-12,rtol=0),'positive preference formula')
                px,py = fractions.copy(),pos['preferences'].copy()
                if px[-1]<1: px.append(1.); py.append(py[-1])
                require(px==pos['auc_fractions'] and py==pos['auc_scores'],'positive-only integration tail')
                require(np.isclose(np.trapz(py,px),row[method+'_positive_pdauc'],atol=1e-12),'secondary PDAUC')
        if (i+1)%50==0: print(f'Validated {i+1}/{n} saved gradient examples',flush=True)
    if n==300:
        require(analyze(rows)==json.loads((root/'summary.json').read_text()),'paired statistics reconstruction')
        a = json.loads((root/'summary.json').read_text())['comparisons']['A_SHAP_objective']['primary']
        old = json.loads((EXP4/'summary.json').read_text())['primary']
        require(a['mean_difference']==old['mean_difference'] and a['wilcoxon']==old['wilcoxon']
                and a['paired_bootstrap_ci95']==old['paired_bootstrap_ci95'],'Experiment4 numerical reuse')
    protected = json.loads((root/'protected_experiments.json').read_text())
    verify_snapshot(protected)
    save_json(root/'verification.json',{'all_passed':True,'n':n,'protected_files_unchanged':len(protected),'zero_relu_maps':zeros,
        'cpu_gpu_float32_absolute_tolerances':{'spatial_weights':2e-6,'maps':1e-6,'region_means_from_saved_pixels':'exact'},
        'scope':'Original sample/captions/SLIC/scores and SHAP preserved; native gradient formula, pre-ReLU linearity, crop projection, masks, crossings, preference, AUC and paired statistics reconstructed; no model inference.'})
    print('Saved gradient evidence verified; Experiments1-4 unchanged',flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory',nargs='?',default='results/experiment_5')
    verify(Path(parser.parse_args().directory))


if __name__=='__main__': main()
