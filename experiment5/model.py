"""Paper's model-preserving multi-head Grad-ECLIP variant.

Source: Zhao et al., ICML 2024, Eq. 7-8 and Appendix C; official notebook
grad_eclip_image.ipynb, commit e370e6cb194faf2020f5d1ed268f9d57e91a38e6.
The authors' demo switches the final block to one head. Here the original twelve
heads remain in the forward pass, as required by the frozen-model experiment.
The notebook's q/k cosine spatial weights and gradient/value formula are kept.
"""
import hashlib

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers.image_transforms import get_resize_output_image_size
from transformers.image_utils import ChannelDimension


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def spatial_weights(q, k):
    # Same concatenated-channel cosine/min-max formula as the official notebook.
    cosine = (F.normalize(q[:, :1], dim=-1)*F.normalize(k[:, 1:], dim=-1)).sum(-1)
    low, high = cosine.min(), cosine.max()
    if not torch.isfinite(cosine).all() or (high-low).item() <= 0:
        raise ValueError('STOP: degenerate Grad-ECLIP spatial weighting')
    return (cosine-low)/(high-low)


def map_from_gradient(gradient, value, spatial):
    signed = (gradient[:, :1]*value[:, 1:]*spatial[..., None]).sum(-1)
    return signed, signed.relu()


def project_map(patch_map, image, processor):
    """Undo processor crop/resize; unseen working-image pixels get zero relevance."""
    ip = processor.image_processor
    if not ip.do_resize or not ip.do_center_crop:
        raise ValueError('Unexpected CLIP processor geometry')
    short = ip.size['shortest_edge']
    resized_h, resized_w = get_resize_output_image_size(image, short, default_to_square=False,
        input_data_format=ChannelDimension.LAST)
    crop_h, crop_w = ip.crop_size['height'], ip.crop_size['width']
    top, left = (resized_h-crop_h)//2, (resized_w-crop_w)//2
    if top < 0 or left < 0:
        raise ValueError('Unexpected padded processor crop')
    crop = F.interpolate(patch_map[None, None], size=(crop_h, crop_w), mode='bilinear', align_corners=False)[0, 0]
    canvas = torch.zeros((resized_h, resized_w), dtype=crop.dtype, device=crop.device)
    canvas[top:top+crop_h, left:left+crop_w] = crop
    pixels = F.interpolate(canvas[None, None], size=image.shape[:2], mode='bilinear', align_corners=False)[0, 0]
    geometry = {'resized_height': resized_h, 'resized_width': resized_w,
        'crop_top': top, 'crop_left': left, 'crop_height': crop_h, 'crop_width': crop_w,
        'working_height': image.shape[0], 'working_width': image.shape[1],
        'interpolation': 'Bilinear align_corners=False; inverse center crop then inverse resize; zero outside visible crop'}
    return pixels.detach().cpu().numpy(), geometry


def region_means(pixels, segments):
    counts = np.bincount(segments.ravel())
    return np.bincount(segments.ravel(), weights=pixels.ravel())/counts


class GradEclip:
    def __init__(self, scorer):
        self.scorer = scorer
        self.model = scorer.model

    def explain(self, image, text, validate=False):
        model = self.model
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Model must be eval and parameter-frozen')
        pixels = self.scorer.processor(images=Image.fromarray(image), return_tensors='pt')['pixel_values'].to(self.scorer.device)
        vision = model.vision_model
        with torch.no_grad():
            x = vision.pre_layrnorm(vision.embeddings(pixels))
            for layer in vision.encoder.layers[:-1]:
                x = layer(x, attention_mask=None, causal_attention_mask=None)[0]
        # Differentiate only the last block, as Grad-ECLIP uses its CLS gradient.
        x = x.detach().requires_grad_(True)
        layer = vision.encoder.layers[-1]
        norm = layer.layer_norm1(x)
        attn = layer.self_attn
        q, k, v = attn.q_proj(norm), attn.k_proj(norm), attn.v_proj(norm)
        batch, tokens, width = q.shape
        heads, head_dim = attn.num_heads, attn.head_dim
        def split(tensor):
            return tensor.view(batch, tokens, heads, head_dim).transpose(1, 2)
        attention = ((split(q)*attn.scale) @ split(k).transpose(-1, -2)).softmax(-1)
        context = (attention @ split(v)).transpose(1, 2).contiguous().reshape(batch, tokens, width)
        hidden = x+attn.out_proj(context)
        hidden = hidden+layer.mlp(layer.layer_norm2(hidden))
        feature = model.visual_projection(vision.post_layernorm(hidden[:, 0]))
        normalized = F.normalize(feature, dim=-1)
        text_tensor = torch.as_tensor(text, dtype=normalized.dtype, device=normalized.device)
        scores = (normalized @ text_tensor.T)[0]
        objectives = [scores[0], scores[0]-scores[1]]
        gradients = [torch.autograd.grad(c, context, retain_graph=True)[0] for c in objectives]
        spatial = spatial_weights(q.detach(), k.detach())
        side = int((tokens-1)**.5)
        if side*side != tokens-1:
            raise ValueError('Non-square CLIP patch grid')
        result = {'cosines': scores.detach().cpu().numpy(), 'patch_side': side,
            'spatial_weights': spatial.detach().cpu().numpy(), 'values': v.detach().cpu().numpy(),
            'queries': q.detach().cpu().numpy(), 'keys': k.detach().cpu().numpy(), 'heads': heads}
        for prefix, gradient in zip(('grad_point', 'grad_cr'), gradients):
            signed, relevance = map_from_gradient(gradient.detach(), v.detach(), spatial)
            patch = relevance[0].reshape(side, side)
            pixel_map, geometry = project_map(patch, image, self.scorer.processor)
            result[prefix] = {'gradient_cls': gradient[:, :1].detach().cpu().numpy(),
                'signed_patch_map': signed[0].reshape(side, side).detach().cpu().numpy(),
                'patch_map': patch.detach().cpu().numpy(), 'pixel_map': pixel_map, 'geometry': geometry}
        result['parameter_gradients_absent'] = all(p.grad is None for p in model.parameters())
        if validate:
            negative = torch.autograd.grad(scores[1], context, retain_graph=True)[0]
            repeated = torch.autograd.grad(objectives[1], context, retain_graph=False)[0]
            error = (gradients[1]-(gradients[0]-negative)).abs().max().item()
            repeat_error = (repeated-gradients[1]).abs().max().item()
            with torch.no_grad():
                native = model.get_image_features(pixel_values=pixels)
                native_scores = (F.normalize(native, dim=-1) @ text_tensor.T)[0]
            result['validation'] = {'native_feature_max_abs_error': (feature.detach()-native).abs().max().item(),
                'native_score_max_abs_error': (scores.detach()-native_scores).abs().max().item(),
                'gradient_linearity_error': error, 'repeat_gradient_error': repeat_error,
                'negative_gradient_cls': negative[:, :1].detach().cpu().numpy()}
        return result
