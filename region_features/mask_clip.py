import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import os
import urllib
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import normalize
from PIL import Image
from pathlib import Path
import math
import itertools

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple = 14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        # maskclip; compute q, k, v
        y = self.ln_1(x)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        L, N, D = y.shape
        y = y.view(L, N, 3, D//3).permute(0, 2, 1, 3).reshape(L, 3*N, D//3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=1)
        v += x
        v = v + self.mlp(self.ln_2(v))

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x, q, k, v

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        q, k, v = None, None, None
        for layer in self.resblocks:
            x, q, k, v = layer(x)
        return x, q, k, v

class MaskCLIP(nn.Module):
    def __init__(self, model_type='ViT-B/32'):
        super().__init__()
        ckpt = self.download(model_type)
        state_dict = ckpt.visual.state_dict()
        width = state_dict["conv1.weight"].shape[0]
        layers = len([k for k in state_dict.keys() if k.endswith(".attn.in_proj_weight")])
        patch_size = state_dict["conv1.weight"].shape[-1]
        grid_size = round((state_dict["positional_embedding"].shape[0] - 1) ** 0.5)
        input_resolution = patch_size * grid_size
        heads = width // 64
        output_dim= ckpt.text_projection.shape[1]
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.grid_size = grid_size
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.preprocess = Compose([
                # T.CenterCrop(224),
                ToTensor(),
                lambda x: x.unsqueeze(0),
                CenterPadding(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        self.load_state_dict(state_dict)

    def download(self, arch):
        url = _MODELS[arch]
        filename = os.path.basename(url)
        download_target = os.path.join(os.path.expanduser("~/.cache/clip"), filename)
        if not os.path.exists(download_target):
            Path(download_target).parent.mkdir(exist_ok=True, parents=True)
            with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
                with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                    while True:
                        buffer = source.read(8192)
                        if not buffer:
                            break

                        output.write(buffer)
                        loop.update(len(buffer))
        # load weights 
        return torch.load(download_target, map_location="cpu")

    def load(self):
        url = _MODELS['ViT-B/32']
        url = _MODELS['ViT-L/14@336px']
        filename = os.path.basename(url)
        download_target = os.path.join(os.path.expanduser("~/.cache/clip"), filename)
        if not os.path.exists(download_target):
            Path(download_target).parent.mkdir(exist_ok=True, parents=True)
            with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
                with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                    while True:
                        buffer = source.read(8192)
                        if not buffer:
                            break

                        output.write(buffer)
                        loop.update(len(buffer))
        # load weights 
        state_dict = torch.load(download_target, map_location="cpu").visual.state_dict()
        self.load_state_dict(state_dict)

    def forward(self, image: Image):
        H, W = image.height, image.width
        with torch.no_grad():
            x = self.preprocess(image).to(self.conv1.weight.data)
            x = self.run(x)
            # x = x.detach().cpu().to(torch.float32).numpy()
            return x

    def forward_with_tensor(self, x):
        B, C, H, W = x.shape
        with torch.no_grad():
            x = normalize(x, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            x = self.run(x)
            return torch.nn.functional.interpolate(x, size=(H, W), mode='bicubic', align_corners=False)

    def run(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        width, pH, pW = x.shape[-3:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # apply pos encoding at varying image size
        # self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        patch_embed = self.positional_embedding[:-1, :].view(self.grid_size, self.grid_size, width).permute(2, 0, 1).to(x.dtype)
        patch_embed = torch.nn.functional.interpolate(patch_embed[None], size=(pH, pW), mode='bicubic', align_corners=False)[0].view(width, -1).permute(1, 0)
        cls_embed = self.positional_embedding[-1:, :].to(x.dtype)
        pos_embed = torch.cat([cls_embed, patch_embed], dim=0)

        x = x + pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, v = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        v = v.permute(1, 0, 2)  # LND -> NLD
        N, L, D = v.shape
        x = self.ln_post(x[:, 1:])
        v = self.ln_post(v[:, 1:])

        if self.proj is not None:
            x = x @ self.proj
            v = v @ self.proj
        dH, dW = H // self.patch_size, W //self.patch_size 

        # print(N, L, D, H, W, dW, dH)
        return v.view(N, L-1, -1).permute(0, 2, 1).view(N, -1, dH, dW)