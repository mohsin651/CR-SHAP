"""Geometry pilot, then grounding scores on frozen annotation targets."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries

from experiment2.io import save_json,sha256,verify_snapshot,write_csv
from experiment2.model import load_image
from .core import analyze,grounding
from .prepare import CONFIG,EXP4,ROOT,source_hashes


def snapshot():
    paths=[]
    for directory in ('lili_shap','experiment2','experiment3','experiment4','experiment5'):
        paths.extend(p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'pdf_tools' not in p.parts)
    paths.extend(p for p in Path('results').rglob('*') if p.is_file() and not any(part.startswith('experiment_6') for part in p.parts))
    paths.extend(Path('.').glob('*.md')); paths.extend(Path('.').glob('requirements*.txt'))
    return {str(p):sha256(p) for p in sorted(paths)}


def overlay(ax,image,record,mask=None):
    height,width=image.shape[:2]; extent=(0,width,height,0)
    ax.imshow(image,extent=extent)
    if mask is not None: ax.imshow(np.ma.masked_where(~mask,mask),cmap='Greens',vmin=0,vmax=1,alpha=.35,extent=extent)
    for x0,y0,x1,y1 in record['transformed_boxes']:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='lime',linewidth=1.6))
    ax.set_xlim(0,width); ax.set_ylim(height,0); ax.axis('off')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=('pilot','full'),default='pilot'); args=parser.parse_args()
    config=json.loads((ROOT/'config.json').read_text())
    if config['experiment']!=CONFIG or config['source_hashes']!=source_hashes(): raise ValueError('Frozen grounding configuration changed')
    for relative,digest in json.loads((ROOT/'target_hashes.json').read_text()).items():
        if sha256(ROOT/relative)!=digest: raise ValueError('Frozen target changed')
    matched=json.loads((ROOT/'matched_manifest.json').read_text())
    protected_path=ROOT/'protected_experiments.json'
    if protected_path.exists(): protected=json.loads(protected_path.read_text())
    else:
        protected=snapshot(); save_json(protected_path,protected)
    verify_snapshot(protected)
    if args.stage=='full':
        review=json.loads((ROOT/'pilot'/'visual_geometry_review.json').read_text())
        if not review['passed'] or review['target_manifest_sha256']!=sha256(ROOT/'matched_manifest.json'):
            raise ValueError('Passed visual geometry review required')
        if (ROOT/'rows.json').exists(): raise FileExistsError(ROOT/'rows.json')
        sample=matched; out=ROOT
    else:
        out=ROOT/'pilot'
        if out.exists(): raise FileExistsError(out)
        out.mkdir(); sample=matched[:5]
        # Add first nontrivial geometry cases, if not already in the fixed prefix.
        for predicate in (lambda r:r['original_dimensions']!=r['working_dimensions'],lambda r:r['exif_orientation']!=1):
            extra=next((r for r in matched if predicate(r)),None)
            if extra and extra not in sample: sample.append(extra)
    (out/'examples').mkdir()
    selected=json.loads((EXP4/'selected_images.json').read_text())
    rows,checks=[] ,[]
    for record in sample:
        i=record['experiment4_index']; source=EXP4/f'example_{i:03d}'
        image=np.asarray(Image.open(source/'original.png')); original=load_image(selected[i]['image_path'])
        if not np.array_equal(image,original): raise ValueError('Working image no longer matches original preprocessing')
        with np.load(source/'raw.npz') as raw:
            segments=raw['segments'].copy(); phis={'point':raw['phi_pointwise'].copy(),'cr':raw['phi_cr_shap'].copy()}
        mask=np.load(ROOT/'targets'/f'{i:03d}'/'gt_union.npz')['mask']
        check={'image_id':record['image_id'],'aligned':segments.shape==mask.shape==image.shape[:2],
            'dimensions':record['working_dimensions']==[image.shape[1],image.shape[0]],
            'valid_target':bool(mask.any())}
        row={'image_id':record['image_id'],'experiment4_index':i,'positive_caption_id':record['positive_caption_id'],
            'positive_caption':record['positive_caption'],'negative_caption':record['negative_caption'],
            'sentence_index':record['sentence_index'],'num_boxes':len(record['transformed_boxes']),
            'gt_union_area_fraction':float(mask.mean())}
        arrays={'segments':segments,'gt_union':mask}
        for prefix,phi in phis.items():
            result,pixels=grounding(phi,segments,mask)
            row.update({prefix+'_'+k:v for k,v in result.items()})
            arrays[prefix+'_phi']=phi; arrays[prefix+'_positive_density']=pixels
            check[prefix+'_mass_conserved']=bool(np.isclose(pixels.sum(),result['eib_denominator'],atol=1e-12,rtol=1e-12))
            check[prefix+'_eib_bounds']=0<=result['eib']<=1+1e-12
        if not all(v for k,v in check.items() if k!='image_id'): raise ValueError(check)
        folder=out/'examples'/f'{i:03d}'; folder.mkdir()
        np.savez_compressed(folder/'grounding.npz',**arrays)
        save_json(folder/'metrics.json',row); save_json(folder/'checks.json',check)
        save_json(folder/'source.json',{'experiment4_directory':str(source),
            'raw_sha256':sha256(source/'raw.npz'),'original_sha256':sha256(source/'original.png'),
            'mapping_sha256':sha256(ROOT/'targets'/f'{i:03d}'/'mapping.json')})
        rows.append(row); checks.append(check)
    save_json(out/'rows.json',rows); write_csv(out/'results.csv',rows)
    save_json(out/'checks.json',{'all_passed':True,'n':len(rows),'examples':checks})
    if args.stage=='pilot':
        fig,axes=plt.subplots(len(sample),2,figsize=(12,4*len(sample)),squeeze=False)
        for axes_pair,record in zip(axes,sample):
            i=record['experiment4_index']; image=np.asarray(Image.open(EXP4/f'example_{i:03d}'/'original.png'))
            raw=np.load(out/'examples'/f'{i:03d}'/'grounding.npz')
            overlay(axes_pair[0],image,record)
            overlay(axes_pair[1],mark_boundaries(image,raw['segments']),record,raw['gt_union'])
            axes_pair[0].set_title(f"{record['image_id']} | raw {record['original_dimensions']} -> working {record['working_dimensions']}")
            axes_pair[1].set_title('Box union + SLIC alignment')
            axes_pair[0].set_xlabel(record['positive_caption']); raw.close()
        fig.tight_layout(); fig.savefig(out/'geometry_overlays.png',dpi=110,bbox_inches='tight'); plt.close(fig)
    else:
        save_json(ROOT/'summary.json',analyze(rows))
        save_json(ROOT/'environment.json',{'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            'no_model_inference':True})
    verify_snapshot(protected)
    print(f'Completed {args.stage}: {len(rows)} grounding examples; prior artifacts unchanged',flush=True)


if __name__=='__main__': main()
