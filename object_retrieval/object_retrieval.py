import sys
import pickle
import json
from typing import List
from pycocotools import mask as mask_utils
# from einops import rearrange, reduce
import gzip
import sklearn
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from scipy.special import softmax, logit
import itertools
import sklearn
import math
import torchvision.transforms as T
import argparse
from tqdm import tqdm
from torch import nn, optim
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
import os
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import object_retrieval_utils as utils 
import clip
from collections import OrderedDict
def load_everything(args):
    if args.query_dict_file != None:
        if os.path.exists(args.query_dict_file) and os.path.exists(args.val_dict_file):
            query_features = utils.open_file(args.query_dict_file)
            val_features = utils.open_file(args.val_dict_file)
    else:
        print('Loading features')
        query_features = OrderedDict()
        val_features = OrderedDict()
        val_labels = utils.open_file(args.val_labels)
        
        for category in tqdm(os.listdir(args.query_dir)):
            category_features = []
            for image in os.listdir(os.path.join(args.query_dir,category)):
                query_image_feature = utils.open_file(os.path.join(args.query_dir,category,image))
                category_features.append(query_image_feature)
            query_features[category] = category_features
        all_val_features = os.listdir(args.val_features)
        
        for val_image in all_val_features:
            val_feature = utils.open_file(os.path.join(args.val_features,val_image))
            
            val_features[val_image] = {'feature':val_feature,'labels':val_labels[val_image.replace('.pkl','')]}
    if args.query_dict_file != None:
        utils.save_file(args.query_dict_file,query_features)
    if args.val_dict_file != None:
        utils.save_file(args.val_dict_file,val_features)
    return query_features, val_features


def compute_single_entry(args,query_features,val_features):
    # query features num_regions x 1024
    # val features list of num_regionsx1024 
    max_region_product = []
    query_features = [entry['region_feature'] for entry in query_features]
    val_features = [entry['region_feature'] for entry in val_features]
    #probably better way to do this but uneven number of regions make it hard 
    stack_query = np.stack(query_features)
    stack_val = np.stack(val_features)
    dot_product = np.dot(stack_query,stack_val.T)
   
    
    max_score = np.max(dot_product.flatten())
    #print(max_score)
    
    return max_score

def compute_dot_product_by_class(args,query_feature_dict,val_feature_dict):
    # 50 query images per category
    # for each image compute, precision@50, mAP
    # average in class
    # average  over class 
    per_class_map =[]
    per_class_precision = []
    for category in tqdm(list(query_feature_dict.keys())):
        cat_map = []
        cat_precision = []
        cat_number = int(category.split('_')[1])
        for query_feature in tqdm(query_feature_dict[category]):
            query_scores = []
            query_labels = []
            
            for val_feature_entry in list(val_feature_dict.keys()):
                
                val_entry = val_feature_dict[val_feature_entry]
                val_feature = val_entry['feature']
              
                val_labels = val_entry['labels'][cat_number]
                
       
                query_scores.append(compute_single_entry(args,query_feature,val_feature))
                if val_labels ==0:
                    query_labels.append(-1)
                else:
                    query_labels.append(val_labels)
            query_ap_score = average_precision_score(np.asarray(query_labels),np.asarray(query_scores))
            # torch metrics needs 0
            query_labels = torch.FloatTensor(query_labels)
            query_labels[query_labels==-1]=0
            query_precision_score = torchmetrics.functional.retrieval.retrieval_precision(torch.FloatTensor(query_scores),torch.FloatTensor(query_labels),top_k=50)
            cat_map.append(query_ap_score)
            cat_precision.append(query_precision_score.item())
    
        
        per_class_map.append(np.mean(cat_map))
        print(f'Average map for category:{np.mean(cat_map)}')
        per_class_precision.append(np.mean(cat_precision))
        print(f'Average class precision for category:{np.mean(cat_precision)}')
    return np.mean(per_class_map), np.mean(per_class_precision)

def object_retrieval(args):
    query_feature_dict, val_feature_dict = load_everything(args)
    per_class_map, per_class_precision = compute_dot_product_by_class(args,query_feature_dict,val_feature_dict)
    print(f'Average map across class:{per_class_map}')
    print(f'Average precision at 50:{per_class_precision}') 
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_dir",
        type=str,
          default=None,
        help="Location of features per region/pooled features",
    )
    parser.add_argument(
        "--query_dict_file",
        type=str,
        default=None,
        help="Location of features per region/pooled features",
    )
    parser.add_argument(
        "--val_dict_file",
        type=str,
        default=None,
        help="Location of features per region/pooled features",
    )
    parser.add_argument(
        "--val_labels",
        type=str,
        default=None,
        help="pkl file of val labels",
    )
    parser.add_argument(
        "--val_features",
        type=str,
         default=None,
        help="dir file of val features",
    )
    args = parser.parse_args()
    load_everything(args)
    object_retrieval(args)
