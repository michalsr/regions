import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_image
from pycocotools import mask as mask_utils
from torchvision.utils import draw_segmentation_masks
import shutil
import region_utils as utils
import torch
from typing import Literal, List, Union

# plt.rcParams["savefig.bbox"] = 'tight'

def show(
    imgs: Union[torch.Tensor,List[torch.Tensor]],
    title: str = None,
    title_y: float = 1,
    subplot_titles: List[str] = None,
    nrows: int = 1,
    fig_kwargs: dict = {}
):
    if not isinstance(imgs, list):
        imgs = [imgs]

    ncols = int(np.ceil(len(imgs) / nrows))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, **fig_kwargs)
    fig.tight_layout()

    for i, ax in enumerate(axs.flatten()):
        if i < len(imgs):
            img = F.to_pil_image(imgs[i].detach().cpu())
            ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            # Set titles for each individual subplot
            if subplot_titles and i < len(subplot_titles):
                ax.set_title(subplot_titles[i])

        else: # Hide subplots with no images
            ax.set_visible(False)

    if title:
        fig.suptitle(title, y=title_y)

    return fig, axs

def get_colors(num_colors, cmap_name='rainbow', as_tuples=False):
    '''
    Returns a mapping from index to color (RGB).

    Args:
        num_colors (int): The number of colors to generate

    Returns:
        torch.Tensor: Mapping from index to color of shape (num_colors, 3).
    '''
    cmap = plt.get_cmap(cmap_name)

    colors = np.stack([
        (255 * np.array(cmap(i))).astype(int)[:3]
        for i in np.linspace(0, 1, num_colors)
    ])

    if as_tuples:
        colors = [tuple(c) for c in colors]

    return colors

def image_from_masks(
    masks: torch.Tensor,
    combine_as_binary_mask: bool = False,
    superimpose_on_image: torch.Tensor = None,
    cmap: str = 'rainbow'
):
    '''
    Creates an image from a set of masks.

    Args:
        masks (torch.Tensor): (num_masks, height, width)
        combine_as_binary_mask (bool): Show all segmentations with the same color, showing where any mask is present. Defaults to False.
        superimpose_on_image (torch.Tensor): The image to use as the background, if provided: (C, height, width). Defaults to None.
        cmap (str, optional): Colormap name to use when coloring the masks. Defaults to 'rainbow'.

    Returns:
        torch.Tensor: Image of shape (C, height, width) with the plotted masks.
    '''
    # Masks should be a tensor of shape (num_masks, height, width)
    if combine_as_binary_mask:
        masks = masks.sum(dim=0, keepdim=True).to(torch.bool)

    # If there is only one mask, ensure we get a visible color
    colors = get_colors(masks.shape[0], cmap_name=cmap, as_tuples=True) if masks.shape[0] > 1 else 'aqua'

    if superimpose_on_image is not None:
        alpha = .8
        background = superimpose_on_image
    else:
        alpha = 1
        background = torch.zeros(3, masks.shape[1], masks.shape[2], dtype=torch.uint8)

    masks = draw_segmentation_masks(background, masks, colors=colors, alpha=alpha)

    return masks

def masks_to_boundaries(masks: torch.Tensor):
    '''
    Given a set of masks, return a set of masks of the boundary pixels.

    masks: (n,h,w)
    Returns: (n,h,w) boolean tensor of boundary pixels
    '''
    # Convert masks to boolean and zero pad the masks to have regions on image edges
    # use the edges as boundaries
    masks = masks.bool()
    masks_padded = torch.nn.functional.pad(masks, (1, 1, 1, 1))

    # Initialize the boundaries tensor with the same size as the padded masks
    boundaries = torch.zeros_like(masks_padded)

    # Compute boundaries, only considering the boundaries of True values
    center = masks_padded[:, 1:-1, 1:-1] # Values in the unpadded image

    boundaries[:, 1:-1, 1:-1] = ( # Assign to original image region
        (center & (center != masks_padded[:, :-2, 1:-1])).float() + # Check if the center is True and the pixel to the left is False
        (center & (center != masks_padded[:, 2:, 1:-1])).float() + # Check if the center is True and the pixel to the right is False
        (center & (center != masks_padded[:, 1:-1, :-2])).float() + # Check if the center is True and the pixel above is False
        (center & (center != masks_padded[:, 1:-1, 2:])).float() # Check if the center is True and the pixel below is False
    ) > 0

    # Remove the padding from the boundaries tensor to match the original size
    boundaries = boundaries[:, 1:-1, 1:-1]

    return boundaries

def compare_model_outputs(
    img_name: str,
    img_dir: str,
    annots_dir: str,
    sam_output_dirs: List[str],
    model_names: List[str],
    cmap='rainbow', # rainbow has a high lightness; see https://matplotlib.org/stable/users/explain/colors/colormaps.html#lightness-of-matplotlib-colormaps
    combine_as_binary_mask: bool = False, # Combine masks and show as masked vs. not-masked region
    superimpose_on_image: bool = True, # Show masks superimposed on image
    save_path: str = None # Where to save the figure
):
    '''
    Compares the segmentation masks of multiple models to the ground truth annotation.

    If combine_as_binary_mask is True, the masks will be combined into a single binary mask indicating
    masked vs. not-masked regions. Otherwise, each mask will be shown separately.

    If superimpose_on_image is True, the masks will be superimposed on the image. Otherwise, the masks
    will be superimposed on a black background. Note that with the rainbow colormap with a highlightness,
    this allows one to distinguish unmasked regions from masked regions.

    Args:
        img_name (str): File name of the image.
        img_dir (str): Directory with all images.
        annots_dir (str): Directory with segmentation annotations (images).
        sam_output_dirs (List[str]): Directories with SAM output JSONs with RLEs.
        model_names (List[str]): Names of the models to display on the figure.
        cmap (str, optional): The colormap to use. Defaults to 'rainbow'.
        combine_as_binary_mask (bool, optional): Combine masks into a single binary mask. Defaults to False.
        superimpose_on_image (bool, optional): Superimpose masks on the original image instead of a black background. Defaults to True.
        save_path (str, optional): Where to save the figure. Defaults to None.
    '''

    img_path = os.path.join(img_dir, img_name)
    image = read_image(img_path)
    superimpose_on_image = image if superimpose_on_image else None

    json_basename = img_name.replace('.jpg','.json')
    output_imgs = []
    for output_dir in sam_output_dirs:
        masks = utils.open_file(os.path.join(output_dir, json_basename))
        masks = [mask_utils.decode(region['segmentation']) for region in masks]
        masks = torch.stack([torch.tensor(m) for m in masks]).to(torch.bool) # (n_masks, h, w)

        image = image_from_masks(masks, combine_as_binary_mask, superimpose_on_image, cmap)
        output_imgs.append(image)

    # Load annotation and convert to a set of masks
    annot_basename = img_name.replace('.jpg','.png')
    annot = read_image(os.path.join(annots_dir, annot_basename))
    annot_masks = torch.cat([annot == val for val in torch.unique(annot)]) # (n_vals, h, w)

    image = image_from_masks(annot_masks, False, superimpose_on_image, cmap)
    output_imgs.append(image)

    fig, _ = show(output_imgs, subplot_titles=model_names + ['Ground Truth'], nrows=2)

    if save_path:
        fig.savefig(save_path, dpi=500)

        # Copy original image
        new_img_path = save_path.split('.')[0] + '_original.jpg'
        shutil.copy(img_path, new_img_path)

def load_everything(img_name: str, img_dir: str, annot_dir: str, sam_output_dir: str):
    '''
    Shows the original image, the combined segmentation masks superimposed on the image,
    something else, then the ground truth annotation.

    Args:
        img_name: basename of the file
        img_dir: dirname to the image
        annot_dir: dirname to the gt annotation?
        sam_output_dir: dirname to the sam output JSON with RLEs
    '''
    image = read_image(os.path.join(img_dir, img_name))
    show(image, title='Original Image')

    json_basename = img_name.replace('.jpg','.json')
    sam_masks = utils.open_file(os.path.join(sam_output_dir, json_basename))
    all_masks = [mask_utils.decode(region['segmentation']) for region in sam_masks]

    combined_masks = torch.tensor(np.stack(all_masks, axis=0).sum(axis=0).astype(bool)) # Binary mask of all masks combined
    superimposed = draw_segmentation_masks(image, combined_masks, colors='aqua')
    show(superimposed, title='Combined Masks') # Visualize masks superimposed on image
    show(combined_masks.to(torch.uint8), title='Combined Masks (binary)') # Visualize in binary "mask/not-mask" colors without background image

    annot_basename = img_name.replace('.jpg','.png')
    annot = read_image(os.path.join(annot_dir, annot_basename))
    show(annot, title='Annotated Masks')

def load_individual_masks(img_name, sam_output_dir, extra=False):
    '''
    Shows first 20 masks in groups of four. If extra is True, shows the next 16 masks in groups of four.

    Args:
        img_name (str): Base name of the image
        sam_output_dir (str): Directory to the sam output JSONs with RLEs.
        extra (bool, optional): Show the next 16 masks, if any. Defaults to False.
    '''
    file_name = img_name.replace('.jpg','.json')
    sam_masks = utils.open_file(os.path.join(sam_output_dir, file_name))
    sorted_regions = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
    all_masks = [mask_utils.decode(region['segmentation']) for region in sorted_regions]

    all_masks_separate = []
    for m in all_masks:
        nonzero = np.nonzero(m)
        m_mask = np.zeros_like(all_masks[0])
        m_mask[nonzero[0],nonzero[1]] = 1
        all_masks_separate.append(torch.from_numpy(m_mask.astype('uint8')))

    for i in range(0, min(20, len(all_masks)), 4):
        show(all_masks_separate[i:i+4], title=f'Masks {i}-{i+3}')

    if extra:
        for i in range(20, min(36, len(all_masks)), 4):
            show(all_masks_separate[i:i+4], title=f'Masks {i}-{i+3}')

