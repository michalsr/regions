'''
Script to generate SLIC superpixels for an image dataset.
'''
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
from PIL import Image
from fast_slic import Slic
from tqdm import tqdm
import torch
import argparse

from sam_analysis.visualize_sam import image_from_masks, show, masks_to_boundaries

def img_from_superpixels(img: torch.tensor, assignment: torch.tensor):
    # On top of the image
    regions = torch.stack([
        assignment == v
        for v in np.unique(assignment)
    ])

    boundaries = masks_to_boundaries(regions)

    overlaid_img = image_from_masks(boundaries, combine_as_binary_mask=True, superimpose_on_image=img)

    return overlaid_img

def run_slic(args):
    os.makedirs(args.output_dir, exist_ok=True)
    basenames = sorted(os.listdir(args.input_dir))
    for filename in tqdm(basenames):
        # Load the image
        with Image.open(os.path.join(args.input_dir, filename)) as f:
            img = np.array(f)

        # Apply the superpixel algorithm
        slic = Slic(num_components=args.num_components, compactness=args.compactness)

        try:
            assignment = slic.iterate(img)
        except ValueError as e:
            print(f'Failed to perform SLIC on {filename}: {e}')
            continue

       
        # Save superpixel data
        ret_dict = {
            'assignment': assignment,
            'clusters': slic.slic_model.clusters
        }

        output_filename = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '.pkl')
        with open(output_filename, 'wb') as f:
            pickle.dump(ret_dict, f)

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str)
    parser.add_argument("--output_dir",type=str)
    parser.add_argument("--num_components",type=int,default=50)
    parser.add_argument("--compactness",type=int,default=8)
    args = parser.parse_args()
    run_slic(args)
