import numpy as np
import pytest

from experiment6.core import caption_tokens,grounding,transform_box,union_mask


def test_caption_matching_is_transport_only_not_semantic():
    assert caption_tokens('A chef’s hat &amp; apron.')==caption_tokens('a chef ’ s hat & apron .')
    assert caption_tokens('A man.')!=caption_tokens('A woman.')
    assert caption_tokens('a part')!=caption_tokens('apart')


def test_inclusive_boxes_scale_clip_and_exif():
    box,_=transform_box([0,0,3,1],(4,2),(8,4))
    assert box==[0.,0.,8.,4.] and union_mask([box],(4,8)).all()
    box,_=transform_box([0,0,1,0],(4,2),(2,4),6)
    assert box==[1.,0.,2.,2.]
    mask=union_mask([box,box],(4,2)); assert mask.sum()==2
    box,meta=transform_box([-1,-1,0,0],(4,2),(4,2))
    assert meta['was_clipped'] and box==[0.,0.,1.,1.]
    with pytest.raises(ValueError): transform_box([2,0,1,1],(4,2),(4,2))


def test_mass_conservation_density_point_and_zero_energy():
    segments=np.array([[0,0,0,1],[0,0,0,1]])
    target=segments==1
    # Larger raw contribution in region0; greater conserved density in region1.
    result,pixels=grounding([3.,2.],segments,target)
    assert np.isclose(pixels.sum(),5.) and np.isclose(pixels[segments==0].sum(),3.)
    assert result['peak_region']==1 and result['point_xy']==[3,0] and result['pointing_game']
    assert result['eib_numerator']==2. and result['eib_denominator']==5. and result['eib']==.4
    zero,pixels=grounding([-1.,0.],segments,target)
    assert zero['zero_positive_energy'] and not zero['pointing_game'] and zero['eib']==0 and zero['point_xy'] is None
    assert not pixels.any()
