# Region Generation 
- Directions for extracting features are under [Step 1: Feature Extraction](#step-1-feature-extraction)
- Directions for generating regions are under [Step 2: Region Generation](#step-2-region-generation). [Part A](#part-a-different-versions-of-sam) describes running different versions of SAM. [Part B](#part-b-slic) describes directions for SLIC.
- Directions for combining SAM masks and image features are under [Step 3: Process Regions](#step-3-process-regions)

## Step 1: Feature Extraction 
`extract_features.py` always requires the image directory `--image_dir` and feature directory as inputs `--feature_dir`. For all models, only features from the last layer were extracted. To modify which layers to use, change `--layers`.   


The default arguments are set for DINOv2 ViT-L/14 so just run:
```
python extract_features.py --image_dir <IMAGE DIRECTORY> --feature_dir <FEATURE DIR>  
```
To run other DINOv2 variants, change the `--multiple` argument and `--model`. To extract more than the just the last layer, change the `--layers`

For DINOv1 ViT-B/16, the layers argument needs to be an integer, not list. Run:
```
python extract_features.py --image_dir <IMAGE DIRECTORY> --feature_dir <FEATURE DIR>  --model_repo_name facebookresearch/dino:main --model dino_vitb16 --layers 1 --multiple 16 
```
For MaskCLIP, run:
```
python extract_features.py --image_dir <IMAGE DIRECTORY> --feature_dir <FEATURE DIR> --model mask_clip --dtype fp32
```
 Too use a different architecture or number of layers for MaskCLIP, the file `mask_clip.py` needs to be changed.

For CLIP ViTB/32, run 
```
python extract_features.py --image_dir <IMAGE DIRECTORY> --feature_dir <FEATURE DIR> --model clip --clip_model ViT-B/32  --dtype fp32 --multiple 32
```
To use a different architecture, change the `--clip_model` and `--multiple` arguments. To use a different number of layers, `extract_clip` function needs to be modified. 

For ImageNet, run
```
python extract_feature.py --image_dir <IMAGE DIRECTORY> --feature_dir <FEATURE DIR> --model imagenet --dtype fp32 --multiple 32
```
To use a different architecture, modify line `292`. The default is ViT-L/32. 

## Step 2: Region Generation 
### Part A: Different Versions of SAM

The relevant checkpoints need to be downloaded:
- For SAM, go to [Original SAM Repo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints). We use `vit_h`. 
- For Mobile-SAM(v1) go to the [Mobile-SAM repo](https://github.com/ChaoningZhang/MobileSAM/tree/master/weights).
- For HQ-SAM, go to [HQ-SAM repo](https://github.com/SysCV/sam-hq#model-checkpoints).We also use `vit_h` for HQ-SAM. 

The `generate_regions.py` is very similar to [SAM's amg.py](https://github.com/facebookresearch/segment-anything/blob/main/scripts/amg.py). There are many hyper-parameters to choose from but we use the default ones. Whenever running `generate_regions.py`, the following arguments have to be provided: an input folder, output folder (to save SAM output) and checkpoint file. We save all SAM output in RLE format, which we use decode in `process_regions.py`

To run SAM ViT-H,
```
python generate_regions.py --input <INPUT FOLDER> --output <OUTPUT FOLDER> --checkpoint <SAM CHECKPOINT> --convert-to-rle
```
To run Mobile-SAM,
```
python generate_regions.py --input <INPUT FOLDER> --output <OUTPUT FOLDER> --checkpoint <MOBILE SAM CHECKPOINT> --convert-to-rle --use_mobile --model_type vit_t 
```
To run HQ-SAM, 
```
python generate_regions.py --input <INPUT FOLDER> --output <OUTPUT FOLDER> --checkpoint <HQ SAM CHECKPOINT> --convert-to-rle --use-hq 
```
### Part B: SLIC 
There are 3 files to run with SLIC:

1) Generate superpixels
```
python slic/gen_superpixels.py --input_dir INPUT DIR --output_dir OUTPUT DIR
```
We used 50 components and 8 for compactness. To change components and compactness, the arguments `--num_components` and `--compactness` can be added. 

2) Convert to SAM format. This converts SLIC output to the same format as SAM. The `--input_dir` should be the `--output_dir` from the previous step.
```
python slic/to_sam_format.py --input_dir <Superpixel output dir> --output_dir OUTPUT DIR
```
3) Merge with SAM. This combines SLIC and SAM. 
```
python slic/merge_sam_and_slic_regions.py --sam_dir SAM DIR --slic_dir SLIC DIR --img_dir IMG DIR --sam_output_dir OUTPUT DIR
```
`--slic_dir` should be the `--output_dir` from step 2. `--sam_dir` should be the `--output` from Part A. 

## Step 3: Process Regions
In `process_regions.py`, we interpolate the image features or the masks and pool within the mask. If using SLIC, make sure that `--mask_dir` contains the merged SLIC and SAM regions. 

To upsample and average pool:
```
python process_regions.py --feature_dir FEATURE DIR --mask_dir MASK DIR --region_feature_dir REGIONS SAVE DIR --interpolate upsample --pooling_method average --dtype bf16
```
To downsample, use `--interpolate downsample` and for max pooling use `--pooling_method max`. If using image features for a model that does not have a patch size of 14, make sure to change the `--dino_patch_length`. Certain image features like CLIP need the dtype to be changed to `--dtype fp32`

## SAM Analysis 
`sam_analysis/visualize_sam.py` has helper functions for visualizing masks on images