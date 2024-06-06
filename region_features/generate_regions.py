import region_utils as utils 
import os
import argparse
import json
import os
import copy
import cv2
from typing import Any, Dict, List
import argparse
from time import time
from tqdm import tqdm

def load_sam_modules(args):
    checkpoint_basename = os.path.basename(args.checkpoint).lower()

    if args.use_hq and args.use_mobile:
        raise ValueError("Cannot have both --use-hq and --use-mobile")

    if args.use_hq:
        try:
            from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator

        except ImportError as e:
            print(e)
            raise ImportError(
                "If segment_anything_hq is not installed, please install it via: pip install segment-anything-hq"
            )

        if args.model_type == 'vit_t':
            args.model_type = 'vit_tiny' # SAM-HQ calls it 'vit_tiny' instead of 'vit_t'

        # Verify that 'sam_hq_vit' is in the checkpoint name
        if not "sam_hq_vit" in checkpoint_basename:
            raise ValueError(
                f"Expected 'sam_hq_vit' in checkpoint name '{checkpoint_basename}'\n"
                + f"Please ensure that the checkpoint is downloaded from: https://github.com/SysCV/sam-hq#model-checkpoints"
            )

    elif args.use_mobile:
        try:
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

        except ImportError as e:
            print(e)
            raise ImportError(
                "If mobile_sam is not installed, please install it via: pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
            )

        # Verify that 'mobile_sam' is in the checkpoint name
        if not "mobile_sam" in checkpoint_basename:
            raise ValueError(
                f"Expected 'mobile_sam' in checkpoint name '{checkpoint_basename}'\n"
                + f"Please ensure that the checkpoint is downloaded from: https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            )

        if not args.model_type == "vit_t":
            print("WARNING: Mobile-SAM uses the 'vit_t' model type; setting --model-type to 'vit_t'")
            args.model_type = "vit_t"

    else: # Default SAM model
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        except ImportError as e:
            print(e)
            raise ImportError(
                "If segment_anything is not installed, please install it via: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        # Verify that 'sam_vit' is in the checkpoint name
        if not "sam_vit" in checkpoint_basename:
            raise ValueError(
                f"Expected 'sam_vit' in checkpoint name '{checkpoint_basename}'\n"
                + f"Please ensure that the checkpoint is downloaded from: https://github.com/facebookresearch/segment-anything"
            )

        if args.model_type == "vit_t":
            raise ValueError(
                "The default SAM library does not support the 'vit_t' model type. "
                + "Please use --use-hq or --use-mobile to use a different model."
            )

    return sam_model_registry, SamAutomaticMaskGenerator

def get_sam_regions(args):
    # basically copy of  segment-anything/scripts/amg.py
    sam_model_registry, SamAutomaticMaskGeneratorCls = load_sam_modules(args)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)

    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGeneratorCls(sam, output_mode=output_mode, **amg_kwargs)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    if args.benchmark:
        targets = targets[:args.num_benchmark_trials]
        print(f'Benchmarking with {len(targets)} trials...')

    gen_times = []
    pbar = tqdm(targets)
    for t in pbar:
        pbar.set_description(f"Processing {t}")

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)

        if os.path.isfile(save_base+".json") and not args.benchmark:
            continue

        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        start_time = time()
        try:
            masks = generator.generate(image)
        except torch.cuda.OutOfMemoryError as e:
            print('Cuda out of memory error')
            torch.cuda.empty_cache()
            continue 
        end_time = time()

        gen_times.append(end_time - start_time)
        if args.benchmark:
            continue

        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=False)
            write_masks_to_folder(masks, save_base)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)

    if len(gen_times) > 0: # May not generate anything if everything already exists
        print(f"Average time per image with {len(gen_times)} trials (seconds): {sum(gen_times)/len(gen_times)}")

    if args.convert_to_rle:
        # add region ids
        sam_files = os.listdir(args.output)
        for f in sam_files:
            new_sam_regions = []
            all_regions = utils.open_file(os.path.join(args.output,f))
            for i,region in enumerate(all_regions):
                image_id = f.replace('.json','')
                region_id = f'{image_id}_region_{i}'
                new_region = copy.deepcopy(region)
                new_region['region_id'] = region_id
                new_sam_regions.append(new_region)

            utils.save_file(os.path.join(args.output, f), new_sam_regions)

    print("Done!")

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #sam regions
    parser.add_argument(
    "--input",
    type=str,
    default=None,
    help="Path to either a single input image or folder of images.")


    parser.add_argument(
    "--output",
    type=str,
    default=None,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ))

    parser.add_argument(
        "--use-hq",
        action="store_true",
        help="Use HQ-SAM model for segmentation."
    )

    parser.add_argument(
        "--use-mobile",
        action="store_true",
        help="Use Mobile-SAM model for segmentation."
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default='vit_h',
        choices=["default", "vit_h", "vit_l", "vit_b", "vit_t"],
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b', 'vit_t']. ",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="The path to the SAM checkpoint to use for mask generation.",
    )

    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    parser.add_argument(
        "--convert-to-rle",
        action="store_true",
        help=(
            "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
            "Requires pycocotools."
        ),
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Evaluate how long it takes on average for the model to generate masks for an image without saving"
            " any output regions. Runs --num-benchmark-trials times.",
        )
    )

    parser.add_argument(
        "--num-benchmark-trials",
        type=int,
        default=100,
        help="The number of times to run mask generation for benchmarking.",
    )

    amg_settings = parser.add_argument_group("AMG Settings")

    amg_settings.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    amg_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    amg_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=None,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    amg_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    amg_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=None,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    amg_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    amg_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    amg_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=None,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    amg_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=None,
        help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
        ),
    )

    args = parser.parse_args()
    get_sam_regions(args)