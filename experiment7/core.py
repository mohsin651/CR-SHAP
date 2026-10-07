import numpy as np

from experiment4.core import analyze,feasible_selection,retrieval_screen
from experiment3.core import BUDGETS,matched_deletion

SELECTION_SEED=2027


def reserve_order(rows,target=200):
    eligible=[r for r in rows if r['original_margin']>0]
    groups={}
    for row in sorted(eligible,key=lambda r:r.get('example_id',r['image_id'])):
        groups.setdefault(row['image_id'],[]).append(row)
    if len(groups)<target: raise ValueError('Insufficient eligible unique images; stop without increasing scope')
    rng=np.random.default_rng(SELECTION_SEED); images=sorted(groups)
    return [groups[images[i]][int(rng.integers(len(groups[images[i]])))] for i in rng.permutation(len(images))]
