import numpy as np
import torch
from PIL import Image, ImageOps

from lili_shap.models import ClipScorer


def load_image(path):
    with Image.open(path) as raw:
        pil = ImageOps.exif_transpose(raw).convert('RGB')
        pil.thumbnail((512, 512), Image.Resampling.LANCZOS)
        return np.array(pil)


class PairClipScorer(ClipScorer):
    """Reuse Experiment 1's frozen model and processor without editing them."""
    @torch.inference_mode()
    def encode_texts(self, captions):
        results = []
        for start in range(0, len(captions), self.batch_size):
            inputs = self.processor(text=captions[start:start+self.batch_size], return_tensors='pt',
                                    padding=True, truncation=True).to(self.device)
            features = self.model.get_text_features(**inputs)
            results.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
        return np.concatenate(results)

    @torch.inference_mode()
    def encode_images(self, images):
        results = []
        for start in range(0, len(images), self.batch_size):
            batch = [Image.fromarray(x) for x in images[start:start+self.batch_size]]
            pixels = self.processor(images=batch, return_tensors='pt')['pixel_values'].to(self.device)
            features = self.model.get_image_features(pixel_values=pixels)
            results.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
        return np.concatenate(results)

    def for_pair(self, positive, negative):
        text = self.encode_texts([positive, negative])
        def score(images):
            # One image embedding per image, shared by both captions and targets.
            return (self.encode_images(images) @ text.T).astype(float)
        return score, text

    @property
    def gamma(self):
        return float(self.model.logit_scale.exp().detach().cpu())
