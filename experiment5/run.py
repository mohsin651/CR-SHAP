"""Run only new gradient explanations on the unmodified Experiment 4 sample."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from scipy.special import expit
import torch

from experiment2.core import preference_deletion
from experiment2.io import save_json, sha256, verify_snapshot, write_csv
from experiment2.screen import configure
from experiment3.core import BUDGETS, matched_deletion
from experiment3.run import COMMIT, PinnedScorer
from .core import analyze
from .model import GradEclip, model_digest, region_means

EXP4 = Path('results/experiment_4')
CONFIG = {'clip_model': 'openai/clip-vit-base-patch32', 'clip_commit': COMMIT,
    'gradeclip_commit': 'e370e6cb194faf2020f5d1ed268f9d57e91a38e6',
    'variant': 'Paper Appendix C multi-head variant preserving original model; notebook cosine spatial weights on concatenated Q/K channels',
    'target_layer': 'Final visual transformer block (index 11 of 12); CLS attention context before output projection',
    'heads': 12, 'patch_grid': [7,7], 'channel_weights': 'Gradient of raw cosine target with respect to CLS pre-projection attention context',
    'spatial_weights': 'Min-max-normalized cosine of unscaled CLS query and patch keys, using concatenated channels as in official notebook',
    'map': 'ReLU(sum_channels(gradient_cls * patch_value * spatial_weight)); no per-map normalization for evaluation',
    'pixel_projection': '7x7 to 224x224 bilinear; inverse actual processor center crop then inverse resize; zero unseen pixels',
    'slic_aggregation': 'Mean relevance over every working-image pixel in each saved Experiment4 SLIC region',
    'primary_endpoint': 'MatchedArea-PDAUC-50', 'secondary_endpoint': 'PositiveOnly-PDAUC on native nonnegative ReLU support',
    'evaluator': 'Unmodified Experiment3 matched_deletion and Experiment2 preference_deletion; Telea radius3, original-image cumulative masks, whole-region first crossing, gamma from original frozen model',
    'bootstrap': {'draws':10000,'seed':42}, 'seed':42, 'sample': 'Exact 300 Experiment4 images and captions; no resampling or replacement',
    'model_updates': 'None; no optimizer; all parameters frozen; autograd.grad on last-block activations only'}


def source_hashes():
    return {str(p):sha256(p) for p in sorted(Path('experiment5').glob('*.py'))}


def snapshot():
    paths = []
    for directory in ('lili_shap','experiment2','experiment3','experiment4'):
        paths.extend(Path(directory).glob('*.py'))
    paths.extend(p for p in Path('results').rglob('*') if p.is_file() and not any(part.startswith('experiment_5') for part in p.parts))
    paths.extend(Path('.').glob('*.md'))
    paths.extend(Path('.').glob('requirements*.txt'))
    for directory in ('data/flickr30k_screening_exp4','data/flickr30k_karpathy_exp4'):
        paths.extend(p for p in Path(directory).rglob('*') if p.is_file())
    return {str(p):sha256(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('pilot','full'),default='pilot')
    args = parser.parse_args()
    output = Path('results/experiment_5_pilot' if args.stage=='pilot' else 'results/experiment_5')
    if output.exists(): raise FileExistsError(output)
    selected = json.loads((EXP4/'selected_images.json').read_text())
    shap_rows = json.loads((EXP4/'rows.json').read_text())
    if len(selected)!=300 or [r['image_id'] for r in selected]!=[r['image_id'] for r in shap_rows]:
        raise ValueError('Experiment4 frozen sample mismatch')
    if not json.loads((EXP4/'verification.json').read_text())['all_passed']:
        raise ValueError('Experiment4 verified artifacts required')
    pilot = Path('results/experiment_5_pilot')
    if args.stage=='full':
        previous = json.loads((pilot/'config.json').read_text())
        if previous['experiment']!=CONFIG or previous['source_hashes']!=source_hashes() or not json.loads((pilot/'verification.json').read_text())['all_passed']:
            raise ValueError('Matching validated pilot required')
        protected = json.loads((pilot/'protected_experiments.json').read_text())
    else:
        print('Snapshotting protected Experiments1-4 and frozen data',flush=True)
        protected = snapshot()
    verify_snapshot(protected)
    output.mkdir(parents=True)
    save_json(output/'protected_experiments.json',protected)
    save_json(output/'selected_images.json',selected)
    save_json(output/'config.json',{'experiment':CONFIG,'stage':args.stage,'source_hashes':source_hashes(),
        'selection_sha256':sha256(EXP4/'selected_images.json'),'shap_results_sha256':sha256(EXP4/'rows.json'),
        'official_notebook_sha256':sha256('experiment5/reference/grad_eclip_image.ipynb')})
    configure()
    scorer = PinnedScorer('cuda' if torch.cuda.is_available() else 'cpu')
    explanation = GradEclip(scorer)
    before = model_digest(scorer.model)
    save_json(output/'environment.json',{'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'device':str(scorer.device),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'model_sha256_before':before})
    gamma = scorer.gamma
    if scorer.model.config._commit_hash!=COMMIT or abs(gamma-100.)>1e-4:
        raise ValueError('Frozen model mismatch')
    rows,checks = [],[]
    if args.stage=='full':
        rows = json.loads((pilot/'rows.json').read_text())
        checks = json.loads((pilot/'checks.json').read_text())['examples']
        for i in range(len(rows)):
            shutil.copytree(pilot/f'example_{i:03d}',output/f'example_{i:03d}')
        save_json(output/'reused_pilot.json',{'n':len(rows),'reason':'Identical source, sample, model and settings; verified maps and metrics reused verbatim.'})
    reused = len(rows)
    sample = selected[:5] if args.stage=='pilot' else selected
    for index,instance in enumerate(sample):
        if index<reused: continue
        print(f'[{index+1}/{len(sample)}] {instance["image_id"]}',flush=True)
        source = EXP4/f'example_{index:03d}'
        image = np.asarray(Image.open(source/'original.png'))
        with np.load(source/'raw.npz') as raw:
            segments,text = raw['segments'].copy(),raw['text_embeddings'].copy()
        row = {k:shap_rows[index][k] for k in ('image_id','positive_caption_id','positive_caption','negative_caption_id',
            'negative_caption','negative_caption_source_image_id','original_positive_cosine','original_negative_cosine','original_margin','original_preference','num_superpixels')}
        for prefix,method in (('point','shap_point'),('cr','shap_cr')):
            row.update({method+k[len(prefix):]:v for k,v in shap_rows[index].items() if k.startswith(prefix+'_')})
        if torch.cuda.is_available(): torch.cuda.synchronize()
        started = time.perf_counter()
        result = explanation.explain(image,text,validate=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        row['shared_gradient_explanation_runtime_seconds'] = time.perf_counter()-started
        check = {'image_id':instance['image_id'],'scores_match_frozen':bool(np.allclose(result['cosines'],
            [row['original_positive_cosine'],row['original_negative_cosine']],atol=1e-6,rtol=0)),
            'negative_owned_elsewhere':row['negative_caption_source_image_id']!=row['image_id'],
            'native_score_match':result['validation']['native_score_max_abs_error']<1e-6,
            'gradient_linearity':result['validation']['gradient_linearity_error']<1e-6,
            'repeat_no_accumulation':result['validation']['repeat_gradient_error']==0.,
            'parameter_gradients_absent':result['parameter_gradients_absent'],
            'model_frozen':not scorer.model.training and all(not p.requires_grad for p in scorer.model.parameters()),
            'expected_grid':result['patch_side']==7 and result['heads']==12}
        cache = {hashlib.sha256(image.tobytes()).hexdigest():np.array([row['original_positive_cosine'],row['original_negative_cosine']])}
        def score(images):
            keys = [hashlib.sha256(i.tobytes()).hexdigest() for i in images]
            missing = {}
            for key,img in zip(keys,images):
                if key not in cache: missing.setdefault(key,img)
            if missing:
                pairs = scorer.encode_images(list(missing.values())) @ text.T
                cache.update(zip(missing,pairs.astype(float)))
            return np.asarray([cache[k] for k in keys])
        matched,positive,masks,arrays = {},{},{},{}
        started = time.perf_counter()
        for method in ('grad_point','grad_cr'):
            data = result[method]
            regions = region_means(data['pixel_map'],segments)
            check[method+'_finite'] = bool(np.isfinite(data['pixel_map']).all() and np.isfinite(regions).all())
            check[method+'_gradient_nonzero'] = bool(np.any(data['gradient_cls']!=0))
            check[method+'_pixel_shape'] = data['pixel_map'].shape==segments.shape
            row[method+'_zero_map'] = bool(not np.any(data['patch_map']>0))
            if args.stage=='pilot' and row[method+'_zero_map']:
                raise ValueError('Technical pilot produced a zero ReLU map; no silent substitute or replacement')
            matched[method],masks[method] = matched_deletion(image,segments,regions,score,gamma)
            positive[method] = preference_deletion(image,segments,regions,score,gamma)
            curve,pos = matched[method],positive[method]
            row.update({method+'_matched_pdauc50':curve['matched_pdauc50'],method+'_positive_pdauc':pos['auc'],
                method+'_num_positive_regions':pos['num_positive_regions'],method+'_positive_area_fraction':pos['positive_area_fraction'],
                method+'_final_positive_preference':pos['preferences'][-1]})
            for q in BUDGETS:
                state = curve['budget_states'][str(q)]
                row.update({f'{method}_preference{q}':curve['preferences'][state], f'{method}_margin{q}':curve['margins'][state],
                    f'{method}_margin_drop{q}':row['original_margin']-curve['margins'][state], f'{method}_rankflip{q}':curve['margins'][state]<0,
                    f'{method}_actual_area{q}':curve['fractions'][state], f'{method}_positive_cosine{q}':curve['positive_cosines'][state],
                    f'{method}_negative_cosine{q}':curve['negative_cosines'][state]})
            for key in ('patch_map','signed_patch_map','pixel_map','gradient_cls'):
                arrays[method+'_'+key] = data[key]
            arrays[method+'_region_scores'] = regions
            arrays[method+'_matched_masks'] = masks[method]
        row['shared_gradient_evaluation_runtime_seconds'] = time.perf_counter()-started
        row['gradient_native_score_error'] = result['validation']['native_score_max_abs_error']
        row['gradient_linearity_error'] = result['validation']['gradient_linearity_error']
        row['gradient_objective_difference'] = row['grad_point_matched_pdauc50']-row['grad_cr_matched_pdauc50']
        if not all(v for k,v in check.items() if k!='image_id'):
            raise ValueError(f'Technical validation failed; no replacement: {check}')
        folder = output/f'example_{index:03d}'
        folder.mkdir()
        np.savez_compressed(folder/'raw.npz',segments=segments,text_embeddings=text,spatial_weights=result['spatial_weights'],
            values=result['values'],queries=result['queries'],keys=result['keys'],
            negative_gradient_cls=result['validation']['negative_gradient_cls'],**arrays)
        save_json(folder/'instance.json',instance)
        save_json(folder/'map_geometry.json',result['grad_point']['geometry'])
        validation = {k:v for k,v in result['validation'].items() if k!='negative_gradient_cls'}
        save_json(folder/'validation.json',validation)
        save_json(folder/'matched_deletion.json',matched)
        save_json(folder/'positive_deletion.json',positive)
        save_json(folder/'checks.json',check)
        rows.append(row); checks.append(check)
        save_json(output/'rows.json',rows); write_csv(output/'results.csv',rows)
    after = model_digest(scorer.model)
    if before!=after: raise ValueError('Model weights changed')
    save_json(output/'model_integrity.json',{'before_sha256':before,'after_sha256':after,'unchanged':True,'gamma':gamma})
    save_json(output/'checks.json',{'all_passed':True,'n':len(rows),'examples':checks})
    if args.stage=='full': save_json(output/'summary.json',analyze(rows))
    verify_snapshot(protected)
    save_json(output/'integrity_check.json',{'all_unchanged':True,'protected_files':len(protected)})
    print(f'Completed {output}; weights and previous experiments unchanged',flush=True)


if __name__=='__main__': main()
