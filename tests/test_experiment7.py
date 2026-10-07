import pytest

from experiment7.core import reserve_order


def test_frozen_unique_image_sampling_and_strict_b16_eligibility():
    rows=[{'image_id':str(i),'example_id':f'{i}:{j}','original_margin':1. if i<220 else 0.}
          for i in range(250) for j in range(2)]
    ordered=reserve_order(rows)
    assert len(ordered)==220 and len({r['image_id'] for r in ordered[:200]})==200
    assert all(r['original_margin']>0 for r in ordered)
    assert ordered==reserve_order(list(reversed(rows)))
    with pytest.raises(ValueError): reserve_order(rows,target=221)

