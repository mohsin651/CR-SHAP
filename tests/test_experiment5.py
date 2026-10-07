import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from experiment5.model import map_from_gradient, project_map, region_means, spatial_weights


def official_functions():
    notebook = json.loads(Path('experiment5/reference/grad_eclip_image.ipynb').read_text(encoding='utf-8'))
    nodes = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            try:
                module = ast.parse(''.join(cell['source']))
            except SyntaxError:
                continue
            nodes.extend(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in ('grad_eclip', 'sim_qk', 'attention_layer'))
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<official-notebook-functions>', 'exec'), namespace)
    return namespace


def test_official_map_formula_and_multihead_attention_equivalence():
    reference = official_functions()
    torch.manual_seed(42)
    q, k, v = [torch.randn(5,1,8) for _ in range(3)]
    context, _ = reference['attention_layer'](q,k,v,num_heads=2)
    context.requires_grad_()
    coefficients = torch.randn(8)
    target = (context[0,0]*coefficients).sum()
    expected = reference['grad_eclip'](target,[q],[k],[v],[context],(2,2))
    gradient = torch.autograd.grad(target,context)[0].permute(1,0,2)
    spatial = spatial_weights(q.permute(1,0,2), k.permute(1,0,2))
    _, observed = map_from_gradient(gradient,v.permute(1,0,2),spatial)
    assert torch.allclose(expected,observed.reshape(2,2),atol=1e-7,rtol=0)
    # The unchanged multi-head forward, rather than the demo's one-head rewrite.
    qq,kk,vv = [t.permute(1,0,2).reshape(1,5,2,4).transpose(1,2) for t in (q,k,v)]
    native = F.scaled_dot_product_attention(qq,kk,vv).transpose(1,2).reshape(1,5,8)
    assert torch.allclose(native,context.permute(1,0,2),atol=1e-6,rtol=0)


def test_crop_inverse_and_mean_region_aggregation():
    processor = SimpleNamespace(image_processor=SimpleNamespace(do_resize=True,do_center_crop=True,
        size={'shortest_edge':224},crop_size={'height':224,'width':224}))
    image = np.zeros((224,448,3),dtype=np.uint8)
    pixels, geometry = project_map(torch.ones(7,7),image,processor)
    assert geometry['crop_left'] == 112 and pixels.shape == (224,448)
    assert np.all(pixels[:,:112] == 0) and np.all(pixels[:,112:336] == 1) and np.all(pixels[:,336:] == 0)
    segments = np.repeat(np.arange(4)[None],112,axis=1).repeat(224,axis=0)
    assert np.array_equal(region_means(pixels,segments),[0,1,1,0])


def test_contrastive_pre_relu_linearity_not_subtracted_positive_maps():
    value = torch.tensor([[[0.,0.],[2.,-1.],[1.,3.]]])
    spatial = torch.tensor([[.5,1.]])
    positive = torch.tensor([[[1.,1.]]])
    negative = torch.tensor([[[2.,-1.]]])
    a, amap = map_from_gradient(positive,value,spatial)
    b, bmap = map_from_gradient(negative,value,spatial)
    c, cmap = map_from_gradient(positive-negative,value,spatial)
    assert torch.equal(c,a-b)
    assert not torch.equal(cmap,amap-bmap)
