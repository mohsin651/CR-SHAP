import html
import re

import numpy as np
from scipy.stats import binomtest

from experiment4.core import paired_statistics


def caption_tokens(text):
    # Exact lexical/punctuation identity after annotation transport normalization.
    return re.findall(r'\w+|[^\w\s]',html.unescape(text).casefold())


def transform_box(box, raw_size, working_size, orientation=1):
    """Official parser gives zero-based inclusive endpoints; return pixel edges."""
    width,height = raw_size; work_width,work_height=working_size
    xmin,ymin,xmax,ymax=map(float,box)
    if not np.isfinite([xmin,ymin,xmax,ymax]).all() or xmax<xmin or ymax<ymin:
        raise ValueError('Invalid/reversed box geometry')
    original=[xmin,ymin,xmax+1,ymax+1]
    clipped=[max(0,xmin),max(0,ymin),min(width,xmax+1),min(height,ymax+1)]
    if clipped[0]>=clipped[2] or clipped[1]>=clipped[3]:
        raise ValueError('Box outside original image')
    points=np.asarray([[clipped[0],clipped[1]],[clipped[2],clipped[1]],
                       [clipped[0],clipped[3]],[clipped[2],clipped[3]]])
    x,y=points.T
    mappings={1:(x,y),2:(width-x,y),3:(width-x,height-y),4:(x,height-y),
        5:(y,x),6:(height-y,x),7:(height-y,width-x),8:(y,width-x)}
    if orientation not in mappings: raise ValueError('Invalid EXIF orientation')
    xx,yy=mappings[orientation]
    oriented_width,oriented_height=(height,width) if orientation>=5 else (width,height)
    xx=xx*work_width/oriented_width; yy=yy*work_height/oriented_height
    transformed=[float(xx.min()),float(yy.min()),float(xx.max()),float(yy.max())]
    return transformed,{'raw_zero_based_inclusive':list(box),'raw_half_open_edges':original,
        'clipped_raw_half_open_edges':clipped,'was_clipped':original!=clipped}


def union_mask(boxes, shape):
    """A pixel belongs iff its center lies in a continuous half-open rectangle."""
    height,width=shape; mask=np.zeros(shape,dtype=bool)
    for xmin,ymin,xmax,ymax in boxes:
        x0,x1=np.clip(np.ceil(np.asarray([xmin,xmax])-.5).astype(int),0,width)
        y0,y1=np.clip(np.ceil(np.asarray([ymin,ymax])-.5).astype(int),0,height)
        mask[y0:y1,x0:x1]=True
    return mask


def grounding(phi, segments, target):
    """Uniform positive mass per region; centroid-nearest peak-density point."""
    phi=np.asarray(phi,dtype=np.float64)
    counts=np.bincount(segments.ravel(),minlength=len(phi))
    if len(counts)!=len(phi) or np.any(counts==0) or target.shape!=segments.shape or not np.isfinite(phi).all():
        raise ValueError('Invalid attribution or aligned geometry')
    positive=np.maximum(phi,0)
    density=positive/counts
    pixels=density[segments]
    denominator=float(positive.sum())
    numerator=float(pixels[target].sum())
    if denominator==0:
        peak_region=None; point=None; hit=False; energy=0.
    else:
        peak_region=int(np.argmax(density))  # First/lowest region index for ties.
        coordinates=np.argwhere(segments==peak_region)
        centroid=coordinates.mean(axis=0)
        distance=((coordinates-centroid)**2).sum(axis=1)
        y,x=coordinates[int(np.argmin(distance))]  # Row-major pixel tie.
        point=[int(x),int(y)]; hit=bool(target[y,x]); energy=numerator/denominator
    return {'pointing_game':hit,'point_xy':point,'peak_region':peak_region,'eib_numerator':numerator,
        'eib_denominator':denominator,'eib':energy,'zero_positive_energy':denominator==0,
        'positive_area_fraction':float((phi[segments]>0).mean()),'num_positive_regions':int((phi>0).sum())},pixels


def analyze(rows):
    if not rows or len({r['image_id'] for r in rows})!=len(rows): raise ValueError('Empty/duplicate grounding sample')
    point=np.asarray([r['point_pointing_game'] for r in rows],bool)
    cr=np.asarray([r['cr_pointing_game'] for r in rows],bool)
    a,b=int((point & ~cr).sum()),int((~point & cr).sum())
    pointing=paired_statistics(rows,'point_pointing_game','cr_pointing_game',True)
    pointing['exact_mcnemar_pvalue']=float(binomtest(a,a+b,.5,alternative='two-sided').pvalue) if a+b else 1.
    pointing['paired_outcomes']={'neither':int((~point & ~cr).sum()),'point_only':a,'cr_only':b,'both':int((point & cr).sum())}
    return {'n':len(rows),'primary':'Pointing Game','secondary':'Energy Inside Box',
        'pointing_game':pointing,'eib':paired_statistics(rows,'point_eib','cr_eib',True),
        'positive_area':paired_statistics(rows,'point_positive_area_fraction','cr_positive_area_fraction',True),
        'gt_union_area_mean':float(np.mean([r['gt_union_area_fraction'] for r in rows])),
        'difference_sign':'CR minus Pointwise for every reported metric; negative area difference means less positive area, not worse grounding.',
        'testing':'Paired exact McNemar for primary Pointing Game; two-sided Wilcoxon for secondary EIB; percentile paired bootstrap10000 seed42; no equivalence inference or multiplicity correction.'}
