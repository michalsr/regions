import pickle
import os
from json import JSONEncoder
from typing import Dict, Optional
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math
import torch
from typing import List, Dict, Any
import logging
import coloredlogs
from PIL import Image, ImageDraw
import warnings
import stat

def save_file(filename,data):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """
    parent_dir = os.path.dirname(filename)
    if parent_dir != '':
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".npy":
        with open(filename, "wb+") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        with open(filename,'w+') as fopen:
            json.dump(data,fopen,indent=2)
    elif file_ext == ".yaml":
        with open(filename, "w+") as fopen:
            dump = yaml.dump(data)
            fopen.write(dump)
            fopen.flush()
    else:
        # assume file is pickle
         with open(filename, "wb+") as fopen:
            pickle.dump(data, fopen)

def open_file(filename):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.txt':
        with open(filename,'r+') as fopen:
            data = fopen.readlines()
    elif file_ext in [".npy",".npz"]:
        data = np.load(filename,allow_pickle=True)
    elif file_ext == '.json':
        with open(filename,'r+') as fopen:
            data = json.load(fopen)
    elif file_ext == ".yaml":
        with open(filename,'r+') as fopen:
            data = yaml.load(fopen,Loader=yaml.FullLoader)
    else:
        # assume pickle
        with open(filename,"rb+") as fopen:
            data = pickle.load(fopen)
    return data
