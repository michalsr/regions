import torch 
import pickle
import os 
import sys
import numpy as np
import json 
from tqdm import tqdm
from pycocotools import mask as mask_utils
from PIL import Image 
import torchvision.transforms as T
import itertools
import math 
import argparse
import gc 
from torch.profiler import profile, record_function, ProfilerActivity
import region_utils as utils  
import torch.nn.functional as F
from pathlib import Path
import clip
from mask_clip import MaskCLIP 
from transformers import ViTFeatureExtractor, ViTModel
import timm
import cv2 
"""
For extraction features for a given dataset and model. 
"""

class FeatureExtractorHook:
    def __init__(self):
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output

    def clear(self):
        self.features = None

def register_hook(model):
    extractor_hook = FeatureExtractorHook()
    hook = model.visual.transformer.register_forward_hook(extractor_hook.hook_fn)
    return hook, extractor_hook

def register_hook1(model):
    extractor_hook = FeatureExtractorHook()
    hook = model.norm.register_forward_hook(extractor_hook.hook_fn)
    return hook, extractor_hook

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
    
def extract_dino_v1(args,model,image):
    layers = eval(args.layers)
    if type(layers) != int:
        raise Exception('DINO v1 only accepts integers for layers. n represents the n last layers used')
    total_block_len = len(model.blocks)
    
    if args.padding != "center":
        raise Exception("Only padding center is implemented")
    transform = T.Compose([
        T.ToTensor(),
        lambda x: x.unsqueeze(0),
        CenterPadding(multiple = args.multiple),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    with torch.inference_mode():
      
        # intermediate layers does not use a norm or go through the very last layer of output
        img = transform(image).to(device='cuda',dtype=args.dtype)
        features_out = model.get_intermediate_layers(img, n=layers)
        features_out = [f.detach().cpu() for f in features_out]
        
        img = img.cpu()
        features = [f[:, 1:] for f in features_out] # Remove the cls tokens
        features = torch.cat(features, dim=-1) # B, H * W, C
        B, _, C= features.size()
        W, H = image.size
        patch_H, patch_W = math.ceil(H / args.multiple), math.ceil(W / args.multiple)
        features = features.permute(0, 2, 1).view(B, C, patch_H, patch_W) 
    return features.to(torch.float32).numpy()

def extract_dino_v2(args,model,image):
    layers = eval(args.layers)
    

    if args.padding != "center":
        raise Exception("Only padding center is implemented")
    transform = T.Compose([
        T.ToTensor(),
        lambda x: x.unsqueeze(0),

        CenterPadding(multiple = args.multiple),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    with torch.inference_mode():
        layers = eval(args.layers)
        # intermediate layers does not use a norm or go through the very last layer of output
        img = transform(image).to(device='cuda',dtype=args.dtype)
        features_out = model.get_intermediate_layers(img, n=layers,reshape=True)    
        features = torch.cat(features_out, dim=1) # B, C, H, W 
    return features.detach().cpu().to(torch.float32).numpy()

def extract_clip(args, model, image, preprocess=None):
    padding_module = CenterPadding(multiple=args.multiple)
    image_input = preprocess(image).unsqueeze(0).to(device=args.device)
    image_input_padded = padding_module(image_input)

    hook, extractor_hook = register_hook(model)  # Register the hook

    with torch.no_grad():
        model.encode_image(image_input_padded)

    hook.remove()  # Remove the hook after use to prevent memory leaks

    if extractor_hook.features is not None:
        features = extractor_hook.features.permute(1, 0, 2)
        sliced_features = features[:, 1:, :]

        ln_post = model.visual.ln_post
        final_features = ln_post(sliced_features)

        return final_features.detach().cpu().to(torch.float32).numpy()
    else:
        raise Exception("Failed to capture features.")
        return None


def extract_mask_clip(args, model, image):
    with torch.no_grad():
        feature = model(image)

    return feature.detach().cpu().to(torch.float32).numpy()[None]
    
def extract_imagenet(args, model, image):
    layers = eval(args.layers)
    
    transform = T.Compose([T.ToTensor(),
                           lambda x: x.unsqueeze(0),
                           CenterPadding(multiple = args.multiple)])
    img = transform(image).to(device='cuda',dtype=args.dtype)
    
    hook, extractor_hook = register_hook1(model)
    with torch.no_grad():
        output = model(img)
        
    hook.remove()
    
    if extractor_hook.features is not None:
        features = [extractor_hook.features]
        features_out = [f.detach().cpu() for f in features]

        img = img.cpu()
        features = [f[:, 1:] for f in features_out] # Remove the cls tokens
        features = torch.cat(features, dim=-1) # B, H * W, C
        B, _, C= features.size()
        W, H = image.size
        patch_H, patch_W = math.ceil(H / args.multiple), math.ceil(W / args.multiple)
        features = features.permute(0, 2, 1).view(B, C, patch_H, patch_W)

    return features.detach().cpu().to(torch.float32).numpy()
    
def extract_features(model, args, preprocess=None):
    all_image_files = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
    Path(args.feature_dir).mkdir(parents=True, exist_ok=True)

    model = model.to(device='cuda', dtype=args.dtype)
    model.eval()

    for i, f in enumerate(tqdm(all_image_files, desc='Extract', total=len(all_image_files))):
        image_name = f

        
        filename_extension = os.path.splitext(image_name)[1]
        try:
            cv = cv2.imread(os.path.join(args.image_dir, f))
            color_coverted = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB) 
            image = Image.fromarray(color_coverted)

        except:
            print(f'Could not read image {f}')
            continue

        if 'dino' in args.model:
            if 'dinov2' in args.model:
                features = extract_dino_v2(args, model, image)
            else:  # dinov1
                features = extract_dino_v1(args, model, image)

        elif args.model == 'clip':
            features = extract_clip(args, model, image, preprocess)
        
        elif args.model == 'mask_clip':
            features = extract_mask_clip(args, model, image)
        
        elif args.model == 'imagenet':
            features = extract_imagenet(args, model, image)

        utils.save_file(os.path.join(args.feature_dir, image_name.replace(filename_extension, ".pkl")), features)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Location of jpg files",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Location to save feature files",
    )
    parser.add_argument(
        "--model_repo_name",
        type=str,
        default="facebookresearch/dinov2",
        choices=['facebookresearch/dinov2','facebookresearch/dino:main'],
        help="PyTorch model name for downloading from PyTorch hub"
    )

    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-L/14@336px"],
        help="CLIP base model version"
    )

    parser.add_argument(
        "--model",
        type=str,
        default='dinov2_vitl14',
        choices=['dinov2_vitl14', 'dino_vitb8', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitg14', 'clip','dino_vitb16','mask_clip', 'imagenet'],  
        help="Name of model from repo"
    )

    parser.add_argument(
        "--layers",
        type=str,
        default="[23]",
        help="List of layers or number of last layers to take"
    )
    parser.add_argument(
        "--padding",
        default="center",
        help="Padding used for transforms"
    )
    parser.add_argument(
        "--multiple",
        type=int,
        default=14,
        help="The patch length of the model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='fp16',
        choices=['fp16', 'fp32','bf16'],
        help="Which mixed precision to use. Use fp32 for clip and Mask_CLIP"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device  

    if args.dtype == "fp16":
      args.dtype = torch.half
    elif args.dtype == "fp32":
      args.dtype = torch.float ## this change is needed for CLIP model
    else: 
      args.dtype = torch.bfloat16

    if args.model == 'clip':
        model, preprocess = clip.load(args.clip_model, device=device)
    elif args.model == 'mask_clip':
        model = MaskCLIP('ViT-L/14@336px').to(device)
    elif args.model == 'imagenet':
        model = timm.create_model('vit_large_patch32_224.orig_in21k', pretrained=True, dynamic_img_size = True)
    else:
        model = torch.hub.load(f'{args.model_repo_name}', f'{args.model}')

    model = model.to(device=args.device, dtype=args.dtype)  

    if args.model == 'clip':
        extract_features(model, args, preprocess) 
    else:
        extract_features(model, args)