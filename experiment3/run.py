import argparse
import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.special import expit
from skimage.segmentation import slic
from transformers import CLIPModel, CLIPProcessor

from experiment2.core import estimate_targets, preference_deletion
from experiment2.io import save_json, sha256, verify_snapshot, write_csv
from experiment2.model import PairClipScorer, load_image
from experiment2.run import CONFIG as EXP2_CONFIG
from experiment2.screen import configure
from lili_shap.core import mean_mask, removal_mask, sample_coalitions
from .core import BUDGETS, analyze, matched_deletion

COMMIT = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
CONFIG = {**EXP2_CONFIG, 'clip_commit':COMMIT, 'selection_seed':2026,
    'matched_evaluation':'All signed SHAP descending with ascending index ties; whole-region first crossing of 10/20/30/40/50 percent; normalized trapezoid to actual 50 crossing; no extrapolation',
    'primary_endpoint':'Positive-only PDAUC, unchanged Experiment 2 definition',
    'runtime':'Shared inference separately from each regression', 'bootstrap':{'resamples':10000,'seed':42,'cluster':'image_id'}}


class PinnedScorer(PairClipScorer):
    def __init__(self, device):
        self.device = torch.device(device)
        self.batch_size = 16
        self.processor = CLIPProcessor.from_pretrained(CONFIG['model'],revision=COMMIT)
        self.model = CLIPModel.from_pretrained(CONFIG['model'],revision=COMMIT).to(self.device).eval()
        self.model.requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['pilot','full'],default='pilot')
    parser.add_argument('--reuse-completed', type=Path)
    args = parser.parse_args()
    screening = Path('data/sugarcrepe_screening_exp3')
    selected_path = screening/'confirmatory_selected_instances.json'
    selected = json.loads(selected_path.read_text())
    preflight_path = screening/'segmentation_preflight.json'
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_text())
        if not preflight['all_passed']:
            raise ValueError(f"STOP: frozen sample cannot support the unchanged coalition protocol: {preflight['failures']}")
    integrity = json.loads((screening/'protected_experiments.json').read_text())
    verify_snapshot(integrity)
    if importlib.metadata.version('scikit-image') != '0.24.0':
        raise ValueError('SLIC version differs')
    if args.stage == 'full':
        pilot = Path('results/experiment_3_pilot')
        previous = json.loads((pilot/'config.json').read_text())
        if not json.loads((pilot/'checks.json').read_text())['all_passed'] or previous['experiment'] != CONFIG or previous['selection_sha256'] != sha256(selected_path):
            raise ValueError('Matching validation pilot required')
    output = Path('results/experiment_3_pilot' if args.stage=='pilot' else 'results/experiment_3')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    configure()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = PinnedScorer(device)
    gamma = scorer.gamma
    if scorer.model.config._commit_hash != COMMIT or gamma != 100.:
        raise ValueError('Model mismatch')
    save_json(output/'config.json',{'experiment':CONFIG,'selection_sha256':sha256(selected_path),'stage':args.stage,'gamma':gamma})
    save_json(output/'environment.json',{'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None})
    for name in ['selection_report.json','screening_summary.json','protected_experiments.json']:
        save_json(output/name,json.loads((screening/name).read_text()))
    save_json(output/'selected_instances.json',selected)
    for name in ['protocol_amendment.json','segmentation_preflight.json']:
        if (screening/name).exists(): save_json(output/name,json.loads((screening/name).read_text()))
    cached = np.load(screening/'text_embeddings.npy')
    captions = {c:i for i,c in enumerate(json.loads((screening/'captions.json').read_text()))}
    rows, checks = [], []
    sample = selected[:7] if args.stage=='pilot' else selected
    if args.reuse_completed:
        if args.stage!='full': raise ValueError('Reuse is only for full runs')
        source = args.reuse_completed
        source_config = json.loads((source/'config.json').read_text())
        if source_config['experiment'] != CONFIG: raise ValueError('Reuse method configuration differs')
        saved_rows = json.loads((source/'rows.json').read_text())
        for old_index,row in enumerate(saved_rows):
            if row['example_id'] != sample[old_index]['example_id']: raise ValueError('Reused prefix differs from amended sample')
            saved_folder = source/f'example_{old_index:03d}'
            saved_check = json.loads((saved_folder/'checks.json').read_text())
            if not all(v for k,v in saved_check.items() if k!='example_id'): raise ValueError('Reused checks failed')
            shutil.copytree(saved_folder,output/f'example_{old_index:03d}')
            rows.append(row)
            checks.append(saved_check)
        write_csv(output/'results.csv',rows)
        save_json(output/'rows.json',rows)
        save_json(output/'reused_prefix.json',{'source':str(source),'n':len(rows),'reason':'Identical frozen configuration and same selected-instance prefix; original saved artifacts already independently verified'})
    reused_count = len(rows)
    for index, instance in enumerate(sample):
        if index < reused_count: continue
        print(f"[{index+1}/{len(sample)}] {instance['example_id']}",flush=True)
        folder = output/f'example_{index:03d}'
        folder.mkdir()
        image = load_image(instance['image_path'])
        Image.fromarray(image).save(folder/'original.png')
        segments = slic(image,n_segments=16,compactness=30,start_label=0,channel_axis=-1)
        _, inverse = np.unique(segments,return_inverse=True)
        segments = inverse.reshape(image.shape[:2])
        d = int(segments.max())+1
        coalitions = sample_coalitions(d,128,42)
        score,text = scorer.for_pair(instance['positive_caption'],instance['negative_caption'])
        original = score([image])[0]
        margin = float(original[0]-original[1])
        if margin <= 0:
            raise ValueError('Selected instance no longer correctly ranked; no replacement allowed')
        if device=='cuda': torch.cuda.synchronize()
        started = time.perf_counter()
        chunks, image_hashes, mask_hashes = [],[],[]
        for offset in range(0,128,16):
            images = []
            for coalition in coalitions[offset:offset+16]:
                mask = removal_mask(segments,coalition)
                assert np.array_equal(mask>0,coalition[segments]==0)
                perturbed = mean_mask(image,mask)
                images.append(perturbed)
                image_hashes.append(hashlib.sha256(perturbed.tobytes()).hexdigest())
                mask_hashes.append(hashlib.sha256(mask.tobytes()).hexdigest())
            chunks.append(score(images))
        pairs = np.concatenate(chunks)
        if device=='cuda': torch.cuda.synchronize()
        shared_seconds = time.perf_counter()-started
        values,fit = estimate_targets(coalitions,pairs)
        check = {'example_id':instance['example_id'],
            'full_matches_original':bool(np.allclose(pairs[1],original,atol=1e-6,rtol=0)),
            'screening_matches':bool(np.allclose(original,[instance['original_positive_cosine'],instance['original_negative_cosine']],atol=1e-6,rtol=0)),
            'text_matches':bool(np.allclose(text,cached[[captions[instance['positive_caption']],captions[instance['negative_caption']]]],atol=1e-6,rtol=0)),
            'frozen':not scorer.model.training and all(not p.requires_grad for p in scorer.model.parameters()),
            'efficiency':all(abs(fit[k]['efficiency_residual'])<1e-8 for k in ['positive','negative','contrastive']),
            'linearity':fit['max_abs_linearity_error']<1e-10}
        if not all(v for k,v in check.items() if k!='example_id'):
            save_json(folder/'checks.json',check)
            raise ValueError(check)
        positive,matched,masks = {},{},{}
        row = {'example_id':instance['example_id'],'image_id':instance['image_id'],'category':instance['sugarcrepe_category'],
            'positive_caption':instance['positive_caption'],'negative_caption':instance['negative_caption'],
            'original_positive_cosine':float(original[0]),'original_negative_cosine':float(original[1]),
            'original_margin':margin,'original_preference':float(expit(gamma*margin)),'num_superpixels':d,
            'max_abs_linearity_error':fit['max_abs_linearity_error'],'shared_inference_runtime_seconds':shared_seconds,
            'point_regression_runtime_seconds':fit['pointwise_fit_seconds'],'cr_regression_runtime_seconds':fit['cr_shap_fit_seconds']}
        for prefix,method in [('point','pointwise'),('cr','cr_shap')]:
            positive[prefix] = preference_deletion(image,segments,values[method],score,gamma)
            matched[prefix],masks[prefix] = matched_deletion(image,segments,values[method],score,gamma)
            curve,full = positive[prefix],matched[prefix]
            order = full['order']
            check[prefix+'_signed_order'] = order==sorted(range(d),key=lambda j:(-values[method][j],j))
            check[prefix+'_initial_scores'] = bool(np.allclose([full['positive_cosines'][0],full['negative_cosines'][0]],original,atol=1e-6,rtol=0))
            for q in BUDGETS:
                state = full['budget_states'][str(q)]
                check[f'{prefix}_crossing{q}'] = full['fractions'][state]>=q/100 and full['fractions'][state-1]<q/100
                row.update({f'{prefix}_preference{q}':full['preferences'][state],f'{prefix}_margin{q}':full['margins'][state],
                    f'{prefix}_margin_drop{q}':margin-full['margins'][state],f'{prefix}_rankflip{q}':full['margins'][state]<0,
                    f'{prefix}_actual_area{q}':full['fractions'][state]})
            row.update({f'{prefix}_positive_pdauc':curve['auc'],f'{prefix}_positive_area_fraction':curve['positive_area_fraction'],
                f'{prefix}_num_positive_regions':curve['num_positive_regions'],f'{prefix}_no_positive_attribution':curve['no_positive_attribution'],
                f'{prefix}_matched_pdauc50':full['matched_pdauc50']})
        if not all(v for k,v in check.items() if k!='example_id'):
            raise ValueError(check)
        row['positive_pdauc_difference'] = row['point_positive_pdauc']-row['cr_positive_pdauc']
        row['matched_pdauc50_difference'] = row['point_matched_pdauc50']-row['cr_matched_pdauc50']
        np.savez_compressed(folder/'raw.npz',segments=segments,coalitions=coalitions,coalition_positive_scores=pairs[:,0],
            coalition_negative_scores=pairs[:,1],phi_pointwise=values['pointwise'],phi_negative=values['negative'],phi_cr_shap=values['cr_shap'],
            text_embeddings=text,perturbation_sha256=np.asarray(image_hashes),mask_sha256=np.asarray(mask_hashes),
            point_matched_masks=masks['point'],cr_matched_masks=masks['cr'])
        save_json(folder/'positive_deletion.json',positive)
        save_json(folder/'matched_deletion.json',matched)
        save_json(folder/'instance.json',{**instance,'source_image_sha256':sha256(instance['image_path'])})
        save_json(folder/'regression.json',fit)
        save_json(folder/'checks.json',check)
        rows.append(row)
        checks.append(check)
        write_csv(output/'results.csv',rows)
        save_json(output/'rows.json',rows)
    save_json(output/'checks.json',{'all_passed':True,'n':len(rows),'examples':checks,'max_abs_linearity_error':max(r['max_abs_linearity_error'] for r in rows)})
    save_json(output/'summary.json',analyze(rows))
    verify_snapshot(integrity)
    save_json(output/'integrity_check.json',{'all_unchanged':True,'n_files':len(integrity)})
    print(f'Completed {output}; all checks passed',flush=True)


if __name__ == '__main__':
    main()
