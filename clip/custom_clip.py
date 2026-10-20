"""CLIP with a learnable text prompt, as used by TPT and AGI-TPT.

Compared with the TPT/C-TPT version this file no longer computes any
regularizer inside the forward pass: the class features T are returned to the
caller (agi_tpt_classification.py), which adds the AGI-TPT loss or a baseline
regularizer (agi/regularizers.py). The context vectors are kept in fp32 so that
AdamW and the manual gradient bookkeeping (override mass, random flip set)
work on fp32 gradients; they are cast to CLIP's dtype inside the prompt.
"""
import torch
import torch.nn as nn

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT = '~/.cache/clip'


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # features of the EOT token (highest id in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """Shared context vectors pi (n_ctx x width) placed before the class name."""

    def __init__(self, clip_model, classnames, n_ctx=4, ctx_init=None):
        super().__init__()
        self.dtype = clip_model.dtype
        self.device = clip_model.visual.conv1.weight.device
        self.token_embedding = clip_model.token_embedding
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial context: "{prompt_prefix}" ({n_ctx} tokens)')

        ctx_vectors = ctx_vectors.float()
        self.ctx_init_state = ctx_vectors.detach().clone()   # pi_0
        self.ctx = nn.Parameter(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        self.n_ctx = n_ctx
        self.reset_classnames(classnames)

    @torch.no_grad()
    def reset(self):
        self.ctx.copy_(self.ctx_init_state)

    def reset_classnames(self, classnames):
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized).type(self.dtype)
        self.token_prefix = embedding[:, :1, :]                  # SOS
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]     # CLS, '.', EOS
        self.tokenized_prompts = tokenized
        self.classnames = classnames
        self.n_cls = len(classnames)

    def forward(self, ctx=None):
        ctx = self.ctx if ctx is None else ctx
        ctx = ctx.to(self.dtype).unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=-2)


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, arch="ViT-B/16", n_ctx=4, ctx_init=None):
        super().__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        self.prompt_learner = PromptLearner(clip, classnames, n_ctx, ctx_init)

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch=None):
        self.prompt_learner.reset_classnames(classnames)

    def get_text_features(self, ctx=None, chunk=None):
        """Unit class features T (N x D, fp32). `chunk` encodes classes in blocks
        (used without gradient for very large N, e.g. ImageNet-21k)."""
        pl = self.prompt_learner
        prompts = pl(ctx)
        if chunk is None:
            t = self.text_encoder(prompts, pl.tokenized_prompts)
        else:
            t = torch.cat([self.text_encoder(prompts[i:i + chunk], pl.tokenized_prompts[i:i + chunk])
                           for i in range(0, prompts.shape[0], chunk)])
        t = t.float()
        return t / t.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_image(self, images):
        f = self.image_encoder(images.type(self.dtype)).float()
        return f / f.norm(dim=-1, keepdim=True)

    def logits(self, image_features, text_features):
        return self.logit_scale.exp().float() * image_features @ text_features.t()

    def forward(self, image):
        return self.logits(self.encode_image(image), self.get_text_features())


def get_coop(clip_arch, classnames, device, n_ctx, ctx_init):
    return ClipTestTimeTuning(device, classnames, arch=clip_arch, n_ctx=n_ctx, ctx_init=ctx_init)
