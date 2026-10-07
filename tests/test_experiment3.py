import numpy as np

from experiment3.core import matched_deletion, paired_statistics, select_sample


def test_signed_ranking_crossings_and_observed_auc():
    image = np.full((10,10,3),100,np.uint8)
    segments = np.repeat(np.arange(4),[1,2,3,4])[:,None]*np.ones((1,10),int)
    def score(images):
        # Deterministic states permit an independent trapezoid check.
        return np.asarray([[.2-.01*i,.1] for i in range(len(images))])
    curve,masks = matched_deletion(image,segments,np.array([0.,-.1,.2,-.1]),score,100.)
    assert curve['order']==[2,0,1,3]
    assert curve['budget_states']=={'10':1,'20':1,'30':1,'40':2,'50':3}
    assert curve['auc_fractions']==[0.,.3,.3,.3,.4,.6]
    assert np.isclose(curve['matched_pdauc50'],np.trapz(curve['auc_preferences'],curve['auc_fractions'])/.6)
    assert masks.shape==(5,10,10)
    assert all(np.isclose((m>0).mean(),x) for m,x in zip(masks,curve['fractions']))


def test_selection_exclusion_unique_images_and_shortage():
    rows = [{'example_id':f'{c}:{i}','image_id':str(i//2),'sugarcrepe_category':c,'original_margin':1.}
            for c in ['add_att','swap_obj'] for i in range(10)]
    selected,report = select_sample(rows,{'add_att:0'},target=4)
    assert len(selected)==8
    assert 'add_att:0' not in {r['example_id'] for r in selected}
    assert all(v['selected_unique_images']==4 for v in report['categories'].values())
    assert selected==select_sample(rows,{'add_att:0'},target=4)[0]
    _,report = select_sample(rows,set(),target=12)
    assert all(v['shortage']==2 for v in report['categories'].values())


def test_cluster_bootstrap_repeats_entire_images():
    rows = [{'image_id':image,'point':value,'cr':0.} for image,value in [('a',1.),('a',3.),('b',10.)]]
    result = paired_statistics(rows,'point','cr')
    # Drawing a twice gives 2; b twice gives 10; a+b gives 14/3.
    ci = result['image_cluster_bootstrap_ci95']
    assert ci['low']==2. and ci['high']==10.
    assert np.isclose(result['mean_difference'],14/3)
    assert result['cr_wins']==3
