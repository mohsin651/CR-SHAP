"""Freeze annotation matching and all spatial targets before reading SHAP values."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps

from experiment2.io import save_json,sha256
from experiment2.model import load_image
from .core import caption_tokens,transform_box,union_mask

DATA=Path('data/flickr30k_entities_exp6')
ROOT=Path('results/experiment_6')
EXP4=Path('results/experiment_4')
COMMIT='68b3d6f12d1d710f96233f6bd2b6de799d6f4e5b'
CONFIG={'annotation_repo':'https://github.com/BryanPlummer/flickr30k_entities','annotation_commit':COMMIT,
    'sample':'Exact300 frozen Experiment4 image/caption instances; annotation-only exclusions; no replacement',
    'caption_match':'Original Karpathy caption ID -> sentence index; exact lexical/punctuation tokens after HTML unescape and Unicode casefold; punctuation whitespace ignored; no fuzzy matching',
    'target':'Union of all explicit valid T+-linked visual boxes; duplicate boxes deduplicated, overlaps counted once; all boxes per phrase retained; notvisual/id0 and phrases without explicit boxes contribute no region',
    'coordinates':'XML and official parser zero-based inclusive endpoints -> half-open pixel edges; clip to raw image; EXIF transform; exact working/raw oriented dimension ratios; no CLIP crop applied to GT',
    'raster':'Pixel centers in continuous half-open boxes; ceil(edge-.5) bounds, clipped to working dimensions',
    'spatial_attribution':'max(phi_region,0)/region_pixel_count per pixel; preserves each positive region contribution and total SHAP mass; same map for Pointing Game and EIB',
    'pointing':'Highest positive density region; lowest region index for tied density; pixel nearest region centroid, row-major y/x for equal distances',
    'zero_energy':'Included as Pointing failure and EIB=0, numerator=denominator=0; explicitly flagged',
    'primary':'Pointing Game','secondary':'EIB','area':'Contextual positive area only, not localization quality',
    'bootstrap':{'resamples':10000,'seed':42},'differences':'CR minus Pointwise throughout; lower positive area is contextual',
    'no_inference':'Reuse original SHAP/SLIC; no CLIP calls, deletion reruns, training, tuning, Grad-ECLIP or subsequent experiments'}


def official_parser():
    spec=importlib.util.spec_from_file_location('flickr30k_entities_official',DATA/'flickr30k_entities_utils.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def source_hashes():
    return {str(p):sha256(p) for p in sorted(Path('experiment6').glob('*.py'))}


def build_record(instance,karpathy,parser):
    image_id=instance['image_id']; sentence_path=DATA/'Sentences'/f'{image_id}.txt'; box_path=DATA/'Annotations'/f'{image_id}.xml'
    if not sentence_path.exists() or not box_path.exists(): raise ValueError('Missing sentence/XML annotation')
    original=karpathy[instance['filename']]
    if original['imgid']!=instance['karpathy_image_id']: raise ValueError('Karpathy image ID mismatch')
    caption_id=int(instance['positive_caption_id'])
    sentence_index=original['sentids'].index(caption_id)
    if original['sentences'][sentence_index]['raw']!=instance['positive_caption']: raise ValueError('Original caption ID/text mismatch')
    sentences=parser.get_sentence_data(str(sentence_path)); annotation=parser.get_annotations(str(box_path))
    if len(sentences)!=5 or sentence_index>=len(sentences): raise ValueError('Unexpected Entities sentence structure')
    matches=[i for i,s in enumerate(sentences) if caption_tokens(s['sentence'])==caption_tokens(instance['positive_caption'])]
    if sentence_index not in matches: raise ValueError('Caption/index does not match normalized exact token identity')
    sentence=sentences[sentence_index]
    with Image.open(instance['image_path']) as raw:
        raw_size=raw.size; orientation=int(raw.getexif().get(274,1)); oriented_size=ImageOps.exif_transpose(raw).size
    if raw_size!=(annotation['width'],annotation['height']): raise ValueError('XML/raw image dimensions mismatch')
    working=load_image(instance['image_path'])
    record={'image_id':image_id,'positive_caption_id':instance['positive_caption_id'],'positive_caption':instance['positive_caption'],
        'negative_caption':instance['negative_caption'],'sentence_index':sentence_index,'entities_sentence':sentence['sentence'],
        'match':'caption ID, original sentence index, normalized exact token identity','phrases':[],
        'original_dimensions':list(raw_size),'oriented_dimensions':list(oriented_size),'working_dimensions':[working.shape[1],working.shape[0]],
        'exif_orientation':orientation,'original_boxes':[],'transformed_boxes':[],
        'sentence_sha256':sha256(sentence_path),'annotation_sha256':sha256(box_path),'image_sha256':sha256(instance['image_path'])}
    seen=set()
    for phrase in sentence['phrases']:
        entry={**phrase,'boxes':[]}; chain=phrase['phrase_id']
        if chain=='0' or 'notvisual' in phrase['phrase_type']:
            entry['status']='nonvisual; no localization target'
        elif chain not in annotation['boxes']:
            entry['status']='scene/no-box annotation' if chain in annotation['scene'] or chain in annotation['nobox'] else 'no explicit box'
        else:
            entry['status']='explicit visual boxes included (including scene-type phrases if explicitly boxed)'
            for box in annotation['boxes'][chain]:
                transformed,geometry=transform_box(box,raw_size,(working.shape[1],working.shape[0]),orientation)
                entry['boxes'].append({**geometry,'transformed_edges':transformed})
                if tuple(box) not in seen:
                    seen.add(tuple(box)); record['original_boxes'].append(box); record['transformed_boxes'].append(transformed)
        record['phrases'].append(entry)
    if not record['transformed_boxes']: raise ValueError('Matched caption has no usable explicit visual boxes')
    mask=union_mask(record['transformed_boxes'],working.shape[:2])
    if not mask.any(): raise ValueError('Transformed box union has no working-image pixel centers')
    record['gt_union_area_fraction']=float(mask.mean())
    return record,mask


def main():
    if ROOT.exists(): raise FileExistsError(ROOT)
    ROOT.mkdir(); (ROOT/'targets').mkdir()
    save_json(ROOT/'config.json',{'experiment':CONFIG,'source_hashes':source_hashes(),
        'frozen_before_outcomes':True,'selected_images_sha256':sha256(EXP4/'selected_images.json'),
        'annotation_archive_sha256':sha256(DATA/'annotations.zip'),'official_parser_sha256':sha256(DATA/'flickr30k_entities_utils.py')})
    parser=official_parser(); selected=json.loads((EXP4/'selected_images.json').read_text())
    karpathy={r['filename']:r for r in json.loads(Path('data/flickr30k_karpathy_exp4/dataset_flickr30k.json').read_text())['images']}
    matched,excluded=[],[]; caption_matches=0
    for index,instance in enumerate(selected):
        try:
            # Count caption identity separately from availability of usable boxes.
            original=karpathy[instance['filename']]; si=original['sentids'].index(int(instance['positive_caption_id']))
            sentences=parser.get_sentence_data(str(DATA/'Sentences'/f"{instance['image_id']}.txt"))
            if caption_tokens(sentences[si]['sentence'])==caption_tokens(instance['positive_caption']): caption_matches+=1
            record,mask=build_record(instance,karpathy,parser); record['experiment4_index']=index
            folder=ROOT/'targets'/f'{index:03d}'; folder.mkdir()
            save_json(folder/'mapping.json',record); np.savez_compressed(folder/'gt_union.npz',mask=mask)
            matched.append(record)
        except (ValueError,KeyError,IndexError,OSError) as error:
            excluded.append({'image_id':instance['image_id'],'experiment4_index':index,'positive_caption_id':instance['positive_caption_id'],
                'positive_caption':instance['positive_caption'],'reason':str(error)})
    save_json(ROOT/'matched_manifest.json',matched); save_json(ROOT/'exclusions.json',excluded)
    save_json(ROOT/'sample_summary.json',{'original_frozen_sample':300,'successfully_caption_matched':caption_matches,
        'grounding_evaluable':len(matched),'excluded':len(excluded),'exclusion_reasons':{reason:sum(r['reason']==reason for r in excluded) for reason in sorted({r['reason'] for r in excluded})},
        'resized_examples':sum(r['original_dimensions']!=r['working_dimensions'] for r in matched),
        'nonidentity_exif_examples':sum(r['exif_orientation']!=1 for r in matched),
        'target_definition_frozen_before_attribution_scoring':True})
    save_json(ROOT/'target_hashes.json',{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'targets').rglob('*') if p.is_file()})
    print(f'Frozen targets: {caption_matches}/300 captions matched; {len(matched)} evaluable; {len(excluded)} annotation exclusions; no attributions scored',flush=True)


if __name__=='__main__': main()
