import torch
from transformers import CLIPModel,CLIPProcessor

from experiment2.model import PairClipScorer
from experiment5.model import model_digest

MODEL='openai/clip-vit-base-patch16'
COMMIT='57c216476eefef5ab752ec549e440a49ae4ae5f3'


class PinnedScorer(PairClipScorer):
    def __init__(self,device):
        self.device=torch.device(device); self.batch_size=16
        self.processor=CLIPProcessor.from_pretrained(MODEL,revision=COMMIT)
        self.model=CLIPModel.from_pretrained(MODEL,revision=COMMIT).to(self.device).eval()
        self.model.requires_grad_(False)
        if self.model.config._commit_hash!=COMMIT or self.model.config.vision_config.patch_size!=16:
            raise ValueError('Incorrect ViT-B/16 model')

