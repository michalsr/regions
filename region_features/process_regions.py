import torch
import os
import numpy as np
import time 
from tqdm import tqdm
from pycocotools import mask as mask_utils
import torch
from PIL import Image 
import torchvision.transforms as T
import math
import os
import argparse
import region_utils as utils
import torch.nn.functional as F
import cv2 
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""
Given extracted regions from SAM, create feature vectors for each region using some method (eg. avg)
"""   

def region_features(args,image_id_to_sam):
    if args.feature_dir!= None:
        features_exist = True 
        # Get the intersection of the feature files and the sam regions
        all_feature_files = [f for f in os.listdir(args.feature_dir) if os.path.isfile(os.path.join(args.feature_dir, f))]
        feature_files_in_sam = [f for f in all_feature_files if os.path.splitext(f)[0] in image_id_to_sam]

        features_minus_sam = set(all_feature_files) - set(feature_files_in_sam)
        if len(features_minus_sam) > 0:
            logger.warning(f'Found {len(features_minus_sam)} feature files that are not in the set of SAM region files: {features_minus_sam}')

    prog_bar = tqdm(feature_files_in_sam)
    def extract_features(f, args,device='cuda',features_exist=True):
        prog_bar.set_description(f'Region features: {f}')

        features = utils.open_file(os.path.join(args.feature_dir,f))

        if len(features.shape)>4:
            features = np.squeeze(features,axis=0)
        file_name = f
        ext = os.path.splitext(f)[1]
        all_region_features_in_image = []
        sam_regions = image_id_to_sam[file_name.replace(ext,'')]

        if args.interpolate == 'downsample':
            f1, h1, w1 = features[0].shape

            for region in sam_regions:
                sam_region_feature = {}
                sam_region_feature['region_id'] = region['instance_id']
                sam_region_feature['area'] = region['area']
                sam_mask = mask_utils.decode(region['segmentation'])
                h2, w2 = sam_mask.shape
                downsampled_mask = torch.from_numpy(sam_mask).cuda()
                downsampled_mask = downsampled_mask.unsqueeze(0).unsqueeze(0)
                downsampled_mask = torch.nn.functional.interpolate(downsampled_mask, size=(h1, w1), mode='nearest').squeeze(0).squeeze(0)

                if torch.sum(downsampled_mask).item() == 0:
                    continue

                features_in_sam = torch.from_numpy(features).cuda().squeeze(dim = 0)[:, downsampled_mask==1].view(f1, -1).mean(1).cpu().numpy()
                sam_region_feature['region_feature'] = features_in_sam
                all_region_features_in_image.append(sam_region_feature)
        else:
            if len(sam_regions) > 0:
                # sam regions within an image all have the same total size
                new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
                patch_length = args.dino_patch_length
                padded_h, padded_w = math.ceil(new_h / patch_length) * patch_length, math.ceil(new_w / patch_length) * patch_length # Get the padded height and width
                upsample_feature = torch.nn.functional.interpolate(torch.from_numpy(features).cuda(), size=[padded_h,padded_w],mode='bilinear') # First interpolate to the padded size
                upsample_feature = T.CenterCrop((new_h, new_w)) (upsample_feature).squeeze(dim = 0) # Apply center cropping to the original size
                f,h,w = upsample_feature.size()

                for region in sam_regions:
                    start_region_time = time.time()
                    sam_region_feature = {}
                    if 'region_id' in region:

                        sam_region_feature['region_id'] = region['region_id']
                    sam_mask = mask_utils.decode(region['segmentation'])
                    if 'area' in region:
                        sam_region_feature['area'] = region['area']
                    r_1, r_2 = np.where(sam_mask == 1)

                    if args.pooling_method == 'average':
                        features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1).mean(1).cpu().numpy()
                    elif args.pooling_method == 'max':
                        input_max, max_indices = torch.max(upsample_feature[:,r_1,r_2].view(f,-1), 1)
                        features_in_sam = input_max.cpu().numpy()

                    sam_region_feature['region_feature'] = features_in_sam
                    all_region_features_in_image.append(sam_region_feature)
                    
        
        utils.save_file(os.path.join(args.region_feature_dir, file_name.replace(ext,'.pkl')), all_region_features_in_image)
    for i,f in enumerate(prog_bar):
        try:
            extract_features(f,args,features_exist=features_exist)

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f'Caught CUDA out of memory error for {f}; falling back to CPU')
            torch.cuda.empty_cache()
            continue 

def load_all_regions(args):
    if len(os.listdir(args.mask_dir)) == 0:
        raise Exception(f"No regions found at {args.mask_dir}")
    logger.info(f"Loading region masks from {args.mask_dir}")
    image_id_to_mask = {}
    for f in tqdm(os.listdir(args.mask_dir)):
        filename_extension = os.path.splitext(f)[1]
        regions = utils.open_file(os.path.join(args.mask_dir,f))
        if not args.use_sam:
            regions = [r for r in regions if 'mask' in list(r.keys())]
        image_id_to_mask[f.replace(filename_extension,'')] = regions
    return image_id_to_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Location of extracted features",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Location of masks (sam or ground truth if given)",
    )

    parser.add_argument(
        "--region_feature_dir",
        type=str,
        default=None,
        help="Location of features per region/pooled features",
    )

    parser.add_argument(
        "--dino_patch_length",
        type=int,
        default=14,
        help="the length of dino patch",
    )

    parser.add_argument(
        "--use_sam",
        action="store_false",
        help="If not using json sam regions"
    )
    
    parser.add_argument(
        "--pooling_method",
        type=str,
        default='average',
        choices=['average', 'max'],
        help='pooling methods'
    )

    parser.add_argument(
        "--interpolate",
        type=str,
        default='upsample',
        choices=['upsample','downsample'],
        help='interpolation'
    )

   
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp16', 'fp32','bf16'],
        help="Which mixed precision to use. Use fp32 for clip and dense_clip"
    )



    args = parser.parse_args()


    image_id_to_mask = load_all_regions(args)
    region_features(args,image_id_to_mask)

    logger.info('Done')

