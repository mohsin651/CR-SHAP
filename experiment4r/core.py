import hashlib
import json

from experiment4.core import analyze,retrieval_screen
from experiment3.core import BUDGETS,matched_deletion


def partition(screening,original_ids):
    eligible=[r for r in screening if r['original_margin']>0]
    ids={r['image_id'] for r in eligible}
    original=set(original_ids)
    if len(eligible)!=793 or len(ids)!=793 or len(original)!=300 or not original<=ids:
        raise ValueError('Not the exact original793/300 population')
    remainder=sorted([r for r in eligible if r['image_id'] not in original],key=lambda r:r['image_id'])
    if len(remainder)!=493 or {r['image_id'] for r in remainder}&original:
        raise ValueError('Invalid original/remainder partition')
    canonical=sorted(ids)
    digest=hashlib.sha256(json.dumps(canonical,separators=(',',':')).encode()).hexdigest()
    return sorted(eligible,key=lambda r:r['image_id']),remainder,digest
