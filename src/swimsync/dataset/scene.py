"""Is this frame a pool shot? Zero-shot CLIP with a few text prompts.

Colour alone can't tell a pool from a cyan stage backdrop or a TV graphic, so the
frame picker asks CLIP how much the frame looks like the "pool" prompts versus the
"other" ones (stage, sofa, empty arena, graphics, ceremony, deck).
On the 400 first picks: removed 56 of 64 non-pool frames, dropped 5 of 336 good ones.
"""

import numpy as np

PROMPTS = {
    "pool": [
        "a swimmer in a swimming pool",
        "legs sticking out of the water in a swimming pool",
        "a swimming pool seen from above with a swimmer",
        "a close-up of a swimmer's face in the water",
        "water splashing in a swimming pool",
    ],
    "other": [
        "a person standing on a stage",
        "people sitting on a sofa",
        "an empty swimming pool arena with spectator seats",
        "a television graphic with a logo",
        "a countdown clock on a screen",
        "people talking at a medal ceremony",
        "an athlete walking on a pool deck",
    ],
}


def pool_probability(logits, kinds):
    """Softmax over all prompts, summed over the "pool" ones."""
    z = np.asarray(logits, float)
    p = np.exp(z - z.max())
    p /= p.sum()
    return float(sum(pi for pi, k in zip(p, kinds) if k == "pool"))


class PoolShotClassifier:
    def __init__(self, device="cpu", arch="ViT-B-32", weights="laion2b_s34b_b79k"):
        import open_clip
        import torch

        self.torch, self.device = torch, device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=weights)
        self.model = self.model.to(device).eval()
        texts = [p for k in PROMPTS for p in PROMPTS[k]]
        self.kinds = [k for k in PROMPTS for _ in PROMPTS[k]]
        with torch.no_grad():
            t = self.model.encode_text(open_clip.get_tokenizer(arch)(texts).to(device))
            self.text = t / t.norm(dim=-1, keepdim=True)

    def score(self, frame_bgr):
        """Probability (0..1) that a BGR frame is a pool shot."""
        from PIL import Image

        img = Image.fromarray(frame_bgr[:, :, ::-1])
        with self.torch.no_grad():
            f = self.model.encode_image(self.preprocess(img).unsqueeze(0).to(self.device))
            f = f / f.norm(dim=-1, keepdim=True)
            logits = (100 * f @ self.text.T)[0].cpu().numpy()
        return pool_probability(logits, self.kinds)
