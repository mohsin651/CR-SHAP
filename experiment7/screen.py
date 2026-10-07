"""Fresh complete B/16 screening of both tasks; no B/32 embedding reuse."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
import torch

from experiment2.data import load_instances
from experiment2.io import save_json,sha256,verify_snapshot,write_csv
from experiment2.model import load_image
from experiment2.screen import configure
from experiment4.screen import distribution,segment
from lili_shap.core import sample_coalitions
from .core import SELECTION_SEED,feasible_selection,reserve_order,retrieval_screen
from .model import COMMIT,MODEL,PinnedScorer,model_digest


def root_for(arm): return Path('data/experiment7_screening')/arm


def snapshot():
    paths=[]
    for directory in ('lili_shap','experiment2','experiment3','experiment4','experiment5','experiment6'):
        paths.extend(p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'pdf_tools' not in p.parts)
    paths.extend(p for p in Path('results').rglob('*') if p.is_file() and not any(part.startswith('experiment_7') for part in p.parts))
    paths.extend(Path('.').glob('*.md')); paths.extend(Path('.').glob('requirements*.txt'))
    return {str(p):sha256(p) for p in sorted(paths)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--arm',choices=('sugarcrepe','flickr30k'),required=True); arm=parser.parse_args().arm
    root=root_for(arm)
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True); protected=snapshot(); save_json(root/'protected_experiments.json',protected)
    if importlib.metadata.version('scikit-image')!='0.24.0': raise ValueError('SLIC version changed')
    configure(); scorer=PinnedScorer('cuda' if torch.cuda.is_available() else 'cpu'); before=model_digest(scorer.model)
    if arm=='sugarcrepe':
        rows=load_instances('data/sugarcrepe')
        if len(rows)!=7511: raise ValueError('Unexpected SugarCrepe task population')
        caption_text=list(dict.fromkeys(t for r in rows for t in (r['positive_caption'],r['negative_caption'])))
        images=list({r['image_id']: {'image_id':r['image_id'],'image_path':r['image_path']} for r in rows}.values())
        captions=[{'caption_id':str(i),'caption':text} for i,text in enumerate(caption_text)]
    else:
        source=Path('data/flickr30k_karpathy_exp4')
        images=json.loads((source/'images.json').read_text()); captions=json.loads((source/'captions.json').read_text())
        if len(images)!=1000 or len(captions)!=5000: raise ValueError('Incorrect Karpathy retrieval split')
        caption_text=[r['caption'] for r in captions]
    started=time.perf_counter(); texts=scorer.encode_texts(caption_text); features=[]
    for offset in range(0,len(images),16):
        features.append(scorer.encode_images([load_image(r['image_path']) for r in images[offset:offset+16]]))
        if offset%320==0: print(f'{arm} screening {min(offset+16,len(images))}/{len(images)}',flush=True)
    features=np.concatenate(features)
    np.save(root/'text_embeddings.npy',texts); np.save(root/'image_embeddings.npy',features)
    save_json(root/'images.json',images); save_json(root/'captions.json',captions)
    if arm=='sugarcrepe':
        text_index={t:i for i,t in enumerate(caption_text)}; image_index={r['image_id']:i for i,r in enumerate(images)}
        retrieval=None
        for row in rows:
            pi,ni=text_index[row['positive_caption']],text_index[row['negative_caption']]
            scores=features[image_index[row['image_id']]] @ texts[[pi,ni]].T
            row.update({'positive_caption_index':pi,'negative_caption_index':ni,
                'positive_caption_id':f"{row['example_id']}:positive",'negative_caption_id':f"{row['example_id']}:negative",
                'negative_caption_source_image_id':'dataset-defined SugarCrepe perturbation',
                'original_positive_cosine':float(scores[0]),'original_negative_cosine':float(scores[1]),
                'original_margin':float(scores[0])-float(scores[1]),'sha256':sha256(row['image_path'])})
    else:
        matrix=features @ texts.T; np.save(root/'similarities.npy',matrix)
        rows,retrieval=retrieval_screen(matrix,images,captions)
    for r in rows:
        r['correctly_ranked']=r['original_margin']>0; r['original_preference']=float(expit(scorer.gamma*r['original_margin']))
    save_json(root/'rows.json',rows); write_csv(root/'screening.csv',rows)
    save_json(root/'eligible_pool.json',[r for r in rows if r['correctly_ranked']])
    summary={'arm':arm,'model':MODEL,'clip_commit':COMMIT,'gamma':scorer.gamma,'screened':len(rows),'unique_images_screened':len(images),
        'correctly_ranked':sum(r['original_margin']>0 for r in rows),'incorrect':sum(r['original_margin']<0 for r in rows),
        'ties':sum(r['original_margin']==0 for r in rows),'distributions':{k:distribution([r[k] for r in rows]) for k in
            ('original_positive_cosine','original_negative_cosine','original_margin')},'retrieval':retrieval,
        'screening_runtime_seconds':time.perf_counter()-started,
        'model_geometry':{'processor_crop':scorer.processor.image_processor.crop_size,'input_size':224,'patch_size':16,'patch_grid':[14,14],'tokens_including_cls':197},
        'model_sha256_before':before,'model_sha256_after':model_digest(scorer.model),'device':str(scorer.device),
        'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        'dataset_source_sha256':sha256('data/sugarcrepe/source.json') if arm=='sugarcrepe' else sha256('data/flickr30k_karpathy_exp4/source.json')}
    if summary['model_sha256_after']!=before: raise ValueError('Model changed during screening')
    if arm=='sugarcrepe':
        summary['category_screening']={c:{'n':sum(r['sugarcrepe_category']==c for r in rows),
            'correct':sum(r['sugarcrepe_category']==c and r['correctly_ranked'] for r in rows)} for c in sorted({r['sugarcrepe_category'] for r in rows})}
    else:
        old=json.loads(Path('data/flickr30k_screening_exp4/rows.json').read_text()); old={r['image_id']:r for r in old}
        both=[r for r in rows if r['correctly_ranked'] and old[r['image_id']]['correctly_ranked']]
        summary['backbone_overlap']={'all1000_same_positive_fraction':float(np.mean([r['positive_caption_id']==old[r['image_id']]['positive_caption_id'] for r in rows])),
            'all1000_same_negative_fraction':float(np.mean([r['negative_caption_id']==old[r['image_id']]['negative_caption_id'] for r in rows])),
            'all1000_both_same_fraction':float(np.mean([r['positive_caption_id']==old[r['image_id']]['positive_caption_id'] and r['negative_caption_id']==old[r['image_id']]['negative_caption_id'] for r in rows])),
            'both_correct':len(both),'b16_only_correct':sum(r['correctly_ranked'] and not old[r['image_id']]['correctly_ranked'] for r in rows),
            'b32_only_correct':sum(not r['correctly_ranked'] and old[r['image_id']]['correctly_ranked'] for r in rows),
            'neither_correct':sum(not r['correctly_ranked'] and not old[r['image_id']]['correctly_ranked'] for r in rows)}
    save_json(root/'screening_summary.json',summary)
    ordered=reserve_order(rows); save_json(root/'seeded_sample_and_reserve.json',ordered); save_json(root/'initial_selected_images.json',ordered[:200])
    (root/'preflight').mkdir(); counts={}; checks=[]
    def check(row,position):
        image=load_image(row['image_path']); labels=segment(image); d=int(labels.max())+1
        counts[row['image_id']]=d; file=root/'preflight'/f"{row['image_id']}.npy"; np.save(file,labels)
        if d>=7: sample_coalitions(d,128,42)
        checks.append({'image_id':row['image_id'],'seeded_position':position,'num_superpixels':d,'feasible':d>=7,
            'segmentation_sha256':sha256(file),'working_image_sha256':__import__('hashlib').sha256(image.tobytes()).hexdigest()})
    for i,row in enumerate(ordered[:200]): check(row,i)
    position=200
    while sum(d>=7 for d in counts.values())<200:
        if position==len(ordered): raise ValueError('Feasible reserve exhausted')
        check(ordered[position],position); position+=1
    _,selected,excluded=feasible_selection(ordered[:position],counts,target=200)
    save_json(root/'technical_exclusions.json',excluded); save_json(root/'selected_images.json',selected)
    save_json(root/'segmentation_preflight.json',{'all_passed':True,'n_final':200,'checks':checks,'before_explanation_outcomes':True})
    save_json(root/'selection_report.json',{'seed':SELECTION_SEED,'n':200,'unique_images':len({r['image_id'] for r in selected}),
        'eligible_instances':sum(r['correctly_ranked'] for r in rows),'eligible_unique_images':len(ordered),'technical_exclusions':len(excluded),
        'rule':'Uniform seeded permutation of eligible unique images; one uniformly selected eligible instance per image; predetermined reserves before technical preflight and outcomes. SugarCrepe is not category-balanced.',
        'categories':{c:sum(r.get('sugarcrepe_category')==c for r in selected) for c in sorted({r.get('sugarcrepe_category','retrieval') for r in selected})}})
    save_json(root/'screening_artifact_hashes.json',{str(p.relative_to(root)):sha256(p) for p in root.iterdir() if p.is_file()})
    verify_snapshot(protected); print(f'{arm}: fresh B/16 screening complete; 200 selected, {len(excluded)} technical exclusions',flush=True)


if __name__=='__main__': main()
