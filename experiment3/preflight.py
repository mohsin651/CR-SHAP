"""Check frozen segmentation supports 128 unique coalitions; never change it."""
import json
from pathlib import Path

import numpy as np
from skimage.segmentation import slic

from experiment2.io import save_json
from experiment2.model import load_image


def main():
    source = Path('data/sugarcrepe_screening_exp3')
    rows = json.loads((source/'confirmatory_selected_instances.json').read_text())
    counts,failures = [],[]
    for i,row in enumerate(rows):
        image = load_image(row['image_path'])
        segments = slic(image,n_segments=16,compactness=30,start_label=0,channel_axis=-1)
        d = len(np.unique(segments))
        record = {'example_id':row['example_id'],'num_superpixels':d,'supports_128_unique_coalitions':d>=7 and 2+2*d<=128}
        counts.append(record)
        if not record['supports_128_unique_coalitions']: failures.append(record)
        if (i+1)%100==0: print(f'Frozen segmentation checked: {i+1}/{len(rows)}',flush=True)
    save_json(source/'segmentation_preflight.json',{'all_passed':not failures,'n':len(rows),'failures':failures,'examples':counts})
    print(json.dumps({'all_passed':not failures,'failures':failures}),flush=True)


if __name__=='__main__':
    main()
