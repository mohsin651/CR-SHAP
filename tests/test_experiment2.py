import numpy as np
import pytest
from scipy.special import expit

from experiment2.core import analyze, estimate_targets, preference_deletion, stratified_select
from lili_shap.core import sample_coalitions


def test_kernel_shap_linearity_for_interacting_games():
    z = sample_coalitions(10)
    pos = .3 + .02*z[:,0] - .03*z[:,1]*z[:,2]
    neg = .2 - .04*z[:,0] + .01*z[:,3]*z[:,4]
    values, fit = estimate_targets(z,np.column_stack([pos,neg]))
    np.testing.assert_allclose(values['cr_shap'],values['pointwise']-values['negative'],atol=1e-12)
    assert fit['max_abs_linearity_error']<1e-12


def test_positive_only_deletion_does_not_treat_tail_as_threshold_state():
    image = np.full((10,10,3),100,dtype=np.uint8)
    segments = np.zeros((10,10),dtype=int)
    segments[:2,:]=1
    def score(images):
        assert len(images)==2
        return np.array([[.3,.2],[.15,.2]])
    curve = preference_deletion(image,segments,[-.2,.5],score,10)
    assert curve['order']==[1]
    np.testing.assert_allclose(curve['fractions'],[0,.2])
    assert curve['state25'] is None and curve['state50'] is None
    assert curve['rankflip25'] is None
    assert curve['auc']==pytest.approx(.2*(expit(1)+expit(-.5))/2 + .8*expit(-.5))


def test_no_positive_attribution_keeps_original_preference_constant():
    image = np.zeros((4,4,3),dtype=np.uint8)
    curve = preference_deletion(image,np.zeros((4,4),dtype=int),[-1],lambda images:np.array([[.4,.3]]),100)
    assert curve['no_positive_attribution']
    assert curve['order']==[] and curve['fractions']==[0]
    assert curve['auc']==pytest.approx(expit(10))
    assert not curve['reached25']


def test_balanced_seeded_selection_excludes_incorrect_without_replacement():
    rows=[{'example_id':f'{c}:{i}','sugarcrepe_category':c,'original_margin':1 if i<20 else -1}
          for c in ['a','b','c','d','e','f','g'] for i in range(30)]
    selected,allocation=stratified_select(rows)
    assert len(selected)==100 and len({r['example_id'] for r in selected})==100
    assert max(allocation.values())-min(allocation.values())==1
    assert all(r['original_margin']>0 for r in selected)
    assert selected==stratified_select(rows)[0]


def test_statistics_use_reached_case_and_common_case_denominators():
    rows=[]
    for i in range(10):
        row={'sugarcrepe_category':'a','pointwise_pdauc':.7,'cr_shap_pdauc':.6,
             'pdauc_difference':.1,'original_margin':float(i)/100}
        for m in ['pointwise','cr_shap']:
            row.update({f'{m}_rankflip25':True if i<5 else None,f'{m}_rankflip50':None,
                        f'{m}_margin25':-.01 if i<5 else None,f'{m}_margin_drop25':.1 if i<5 else None,
                        f'{m}_num_positive_regions':3})
        rows.append(row)
    result=analyze(rows)
    rates=result['overall']['rankflip']['25']
    assert rates['pointwise']['percent_among_reached']==100
    assert rates['pointwise']['percent_of_all_with_observed_flip']==50
    assert rates['paired_common_coverage']['n']==5
    assert result['overall']['paired_bootstrap_mean_difference_ci95']['low']==pytest.approx(.1)
