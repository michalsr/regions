# Download Data 
- Download ADE20K from [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
- Download PascalVOC from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html#devkit). The original dataset does not separate training images and annotations into train/val but for training and eval the two are separated. The file `VOCdevkit/VOC2012/ImageSets/Segmentation` contains the train/val split. 
 
 # Generate region features
 - Follow the directions in [Region_Generation_README.md](../region_generation/Region_Generation_README.md) for extracting features, generating regions and processing regions. Note that some images in ADE20K take too much memory on an A40 during feature extraction. We skipped those images during training. 
 
 # Label Regions
 - Using generated regions and ground truth data, run `label_regions.py` for both training and validation sets. Assume that generated regions are in <REGION DIR>,  that region labels will be stored in <REGION LABELS>, and that original labels are in <ANNOTATION DIR>, run the following for the ADE dataset:
 ```
 python label_regions.py --region_labels <REGION LABELS> --annotation_dir <ANNOTATION DIR> --sam_dir <REGION DIR> --num_classes 151
 ```
 For PascalVOC, the command is the same excep that `--num_classes` should be updated to 21.

  # Training 
The following need to be defined for training:
- `--train_region_labels_dir` and `--val_region_labels_dir`, which contain the generated region labels from the step above.
- `--train_region_feature_dir` and `--val_region_feature_dir` which contain generated region features from the [Region Generation Folder](../region_generation/). 
- `--save_dir` store the model output and results.
- `--sam_dir` is the directory for generated **validation** regions (i.e. SAM val regions). This is used for evaluation.
- `--annotation_dir` is the directory for **validation** annotations 

At the beginnig of training and evaluation, we load all of the data into memory by reading the data from all the fiels. There is the option to save the loaded data to a pkl file called `--train_data_file` and `--val_data_file`. 

Currently, `segmentation_training.py` will throw an error if the number of classes is not equal to 150 or 21. 

The mIOU is computed `--iou_every` epochs. Note that this takes longer than training or evaluating an epoch. 

To train on ADE, 
```
python segmentation_training.py --train_region_labels_dir <TRAIN REGION LABELS> --val_region_labels_dir <VAL REGION LABELS> --epochs 15 --train_region_feature_dir <TRAIN REGION FEATURE DIR> --val_region_feature_dir <VAL REGION FEATURE DIR> --save_dir <SAVE DIR> --lr 5e-4 --sam_dir <SAM DIR> --annotation_dir <ANNOTATION DIR> --num_classes 150 --use_scheduler
```
