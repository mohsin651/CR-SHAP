from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ClipScorer:
    def __init__(self, device, batch_size=16):
        self.device = torch.device(device)
        self.batch_size = batch_size
        name = 'openai/clip-vit-base-patch32'
        self.processor = CLIPProcessor.from_pretrained(name)
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def for_caption(self, caption):
        inputs = self.processor(text=[caption], return_tensors='pt', padding=True, truncation=True).to(self.device)
        text = self.model.get_text_features(**inputs)
        text = torch.nn.functional.normalize(text, dim=-1)

        @torch.inference_mode()
        def score(images):
            result = []
            for start in range(0, len(images), self.batch_size):
                batch = [Image.fromarray(x) for x in images[start:start+self.batch_size]]
                pixels = self.processor(images=batch, return_tensors='pt')['pixel_values'].to(self.device)
                features = self.model.get_image_features(pixel_values=pixels)
                features = torch.nn.functional.normalize(features, dim=-1)
                result.extend((features @ text.T).squeeze(1).cpu().numpy().tolist())
            return np.asarray(result)
        return score


class LamaInpainter:
    """Pretrained big-LaMa TorchScript; explicit padding and outside-mask preservation."""
    URL = 'https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt'

    def __init__(self, device, weights=None):
        self.device = torch.device(device)
        path = Path(weights) if weights else Path(torch.hub.get_dir()) / 'checkpoints' / 'big-lama.pt'
        if not path.exists():
            if weights:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(self.URL, str(path))
        self.weights_path = path.resolve()
        self.model = torch.jit.load(str(path), map_location=self.device).eval()

    @torch.inference_mode()
    def __call__(self, image, mask):
        if not mask.any():
            return image.copy()
        h, w = mask.shape
        ph, pw = (-h) % 8, (-w) % 8
        padded = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode='symmetric')
        padded_mask = np.pad(mask, ((0, ph), (0, pw)), mode='symmetric')
        x = torch.from_numpy(padded.copy()).permute(2, 0, 1).float()[None].to(self.device) / 255
        m = torch.from_numpy(padded_mask.copy()).float()[None, None].to(self.device) / 255
        # TorchScript's profiling executor changes numerics after the first
        # call for a new shape. The same mask must define the same SHAP game.
        with torch.jit.optimized_execution(False):
            output = self.model(x, m)
        if not torch.isfinite(output).all():
            raise ValueError('Non-finite LaMa output')
        raw = output[0].permute(1, 2, 0).cpu().numpy()[:h, :w]
        if raw.shape != image.shape or raw.min() < -0.05 or raw.max() > 1.05:
            raise ValueError(f'Unexpected LaMa dimensions/range: {raw.shape}, {raw.min()}, {raw.max()}')
        result = np.clip(raw * 255, 0, 255).astype(np.uint8)
        result[mask == 0] = image[mask == 0]
        return result
