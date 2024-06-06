# Generating Features 
- Region features using ground truth masks for queries using DINOv2 are located in `coco_query_mask.tar.gz`. 
- To generate features for COCO validation, follow the steps in [Region Features](../region_features/) with the COCO validation dataset. 
- The labels used are in `coco_val_retrieval.pkl`. Each image has an array where a 1 indicates that an object of that category is present in the image. 

To evaluate object retrieval, run the following:
```
python object_retrieval.py --query_dir coco_query_masks/ --val_labels coco_val_retrieval.pkl --val_features <VAL FEAT DIR>
```
