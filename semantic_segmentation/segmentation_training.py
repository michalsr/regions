import sys
import pickle
import json
from typing import List
from pycocotools import mask as mask_utils
import torch
import einops
import numpy as np
import torch.nn.functional as F
from segmentation_utils import mean_iou
from PIL import Image
import yaml
import itertools
import math
import argparse
from tqdm import tqdm
from torch import nn, optim
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import segmentation_utils as utils
torch.manual_seed(0) 
import torchvision
import logging 
import coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger)


def get_all_features(region_feat_dir, region_labels_dir,data_file=None):
    if data_file != None:
        if os.path.exists(data_file):
            data = utils.open_file(data_file)
            return np.stack(data['features']),np.stack(data['labels']),np.stack(data['weight'])

    all_feats = []
    all_labels = []
    all_weight = []
    logger.info('Loading features')
    for file_name in tqdm(os.listdir(region_feat_dir)):
        region_feats = utils.open_file(os.path.join(region_feat_dir,file_name))
        labels = utils.open_file(os.path.join(region_labels_dir,file_name))
        
        for i,region in enumerate(region_feats):
            area_feature = region['region_feature']
            area_label = labels[i]['labels']
            area_weight = region['area']
            target_label = list(area_label.keys())[0]

            if area_label[target_label] == 1:

                all_feats.append(area_feature)

                all_labels.append(target_label)
                all_weight.append(area_weight)
    if data_file != None:
        utils.save_file(data_file,{'features':all_feats,'labels':all_labels,'weight':all_weight})
    return np.stack(all_feats), np.stack(all_labels),np.stack(all_weight)

class FeatureDataset(Dataset):
    def __init__(self,region_feat_dir, region_labels_dir,data_file=None):
        super().__init__()
        region_feats,region_labels,weight = get_all_features(region_feat_dir, region_labels_dir,data_file)
        self.region_feats = region_feats
        self.labels = region_labels
        self.weight = weight


    def __len__(self):
        return len(self.region_feats)

    def __getitem__(self, idx):
        region_feats = self.region_feats[idx]
        labels = self.labels[idx]
        weight = self.weight[idx]
        return torch.tensor(region_feats), torch.tensor(labels), torch.tensor(weight)
def eval_acc(args,model,epoch):
    dataset = FeatureDataset(region_feat_dir=args.val_region_feature_dir,region_labels_dir=args.val_region_labels_dir,data_file=args.val_data_file)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

    if args.ade:
        criterion = nn.CrossEntropyLoss(reduction='sum',ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum')
    mca = MulticlassAccuracy(num_classes=args.num_classes+1, average='micro',top_k=1)
    predictions = []
    all_labels = []
    total_regions = 0
    all_loss = 0
    model.eval()
    logger.info('Beginning validation')
    with torch.no_grad():

        for i, data in enumerate(tqdm(dataloader)):
            region_feats, labels, weight= data
            total_regions += len(labels)
            model = model.cuda()

            labels = labels.cuda()
            region_feats = region_feats.cuda()
            outputs = model(region_feats.float())

            outputs = outputs.squeeze()


            # Reshape outputs and labels for loss calculation
            outputs = outputs.view(-1, args.num_classes+1)
            predictions.append(outputs.cpu())
            labels = labels.view(-1)
            all_labels.append(labels.cpu())

            loss = criterion(outputs, labels)
            all_loss+=(loss.item())


    val_loss = all_loss/total_regions
    logger.info(f'Val loss:{val_loss}')
    predictions = torch.stack(predictions)
    all_labels = torch.stack(all_labels)
    val_acc = mca(predictions.squeeze(),all_labels.squeeze())
    logger.info(f'Val acc:{val_acc.item()}')
    return val_loss,val_acc.item()


def train_model(args):
    dataset = FeatureDataset(region_feat_dir=args.train_region_feature_dir,region_labels_dir=args.train_region_labels_dir,data_file=args.train_data_file)
    if args.model == 'linear':
        model = torch.nn.Linear(args.input_channels,args.num_classes+1)
    else:
        model = torchvision.ops.MLP(in_channels=args.input_channels,hidden_channels=[args.hidden_channels,args.num_classes+1])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    if args.ade:
        criterion = nn.CrossEntropyLoss(reduction='none',ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
    epochs = args.epochs
    mca = MulticlassAccuracy(num_classes=args.num_classes+1, average='micro',top_k=1)
    # batch is over total number of regions so can make it very large (8192)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    logger.info(f'Train dataloader length with batch size {args.batch_size}: {len(dataloader)}')
    train_outputs = []
    train_labels = []
    total_regions = 0
    best_miou = -1 
    for epoch in range(epochs):  # Example number of epochs
        logger.info(f'Beginning Training for epoch:{epoch}')
        batch_loss = 0
        model = model.cuda()
        model.train()
        num_regions = 0
        train_acc = 0
        for i, data in enumerate(tqdm(dataloader)):
            model.train()
            region_feats, labels,weight = data

            
            region_feats = region_feats.cuda()
            labels = labels.cuda()

            outputs = model(region_feats.float())
            outputs = outputs.squeeze()

            outputs = outputs.view(-1, args.num_classes+1)

            labels = labels.view(-1)
            num_regions += labels.size()[0]
            weight = weight.cuda()
            weight = torch.nn.functional.normalize(weight.float(),dim=0)


            loss = (criterion(outputs, labels)*weight).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():

                train_acc += (mca(outputs.cpu(),labels.cpu()).item() * labels.size()[0])



        logger.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        torch.save(model.cpu().state_dict(),os.path.join(args.save_dir,'model.pt'))
        val_loss,val_acc = eval_acc(args,model,epoch)
        train_acc = train_acc/num_regions
        logger.info(f"Train_acc:{train_acc}")
        metrics = {'val_loss':val_loss,'val_acc':val_acc,'train_acc':train_acc,'train_loss':loss.item()}
        utils.save_file(os.path.join(args.save_dir,'results',f'metrics_epoch_{epoch}.json'),metrics)
        if args.use_scheduler:
            scheduler.step()

        if args.log_to_wandb:
            metrics.update({
                'epoch': epoch,
                'lr': scheduler.get_last_lr()[0]
            })
            

        if (epoch+1)%args.iou_every==0:
            logger.info('Computing mIOU')
            all_pixel_predictions, file_names = eval_model(args)

            miou = compute_iou(args,all_pixel_predictions,file_names,epoch)
            metrics['mean_iou'] = miou['mean_iou']
            if miou['mean_iou'] >best_miou:
                best_miou = miou['mean_iou']
                torch.save(model.cpu().state_dict(),os.path.join(args.save_dir,'model_best.pt'))
        wandb.log(metrics)
def compute_iou(args,predictions,file_names,epoch):
    actual_labels = []
    for file in tqdm(file_names):
        actual = np.array(Image.open(os.path.join(args.annotation_dir,file.replace('.pkl','.png'))))
        actual_labels.append(actual)

    # Handle predictions where there were no regions
    predictions = [np.full(actual.shape, 255) if p is None else p for p, actual in zip(predictions, actual_labels)]

    if args.ignore_zero or args.ade:
        num_classes = args.num_classes-1
        reduce_labels = True
        reduce_pred_labels=True

    else:
        num_classes = args.num_classes+1
        reduce_labels = False
        reduce_pred_labels=False
    if args.ade==True:
        assert reduce_labels==True
        assert reduce_pred_labels==True
        assert num_classes == 149
    miou = mean_iou(results=predictions,gt_seg_maps=actual_labels,num_labels=num_classes,ignore_index=255,reduce_labels=reduce_labels,reduce_pred_labels=reduce_pred_labels)
    logger.info(miou)
    miou['per_category_iou'] = miou['per_category_iou'].tolist()
    miou['per_category_accuracy'] = miou['per_category_accuracy'].tolist()
    utils.save_file(os.path.join(args.save_dir,'results',f'mean_iou_epoch_{epoch}.json'),miou)
    return miou

def eval_model(args):
    if args.model == 'linear':
        model = torch.nn.Linear(args.input_channels, args.num_classes+1)
    else:
        model = torchvision.ops.MLP(in_channels=args.input_channels,hidden_channels=[args.hidden_channels,args.num_classes+1])

    model.load_state_dict(torch.load(os.path.join(args.save_dir,'model.pt')))
    model.eval()
    class_preds = []
    model = model.cuda()
    all_pixel_predictions = []
    # keep track of order of predictions
    file_names = []
    val_features = args.val_region_feature_dir
    softmax = torch.nn.Softmax(dim=1)
    val_files = [filename for filename in os.listdir(val_features)]
    for file in tqdm(val_files):
        file_names.append(file)
        all_sam = utils.open_file(os.path.join(args.sam_dir,file.replace('.pkl','.json')))

        

        all_regions = []
        region_order = []
        region_features = utils.open_file(os.path.join(val_features,file))
        feature_all = []
        for j,area in enumerate(region_features):

            features = area['region_feature']
            feature_all.append(features)

        region_all = {area['region_id']:j for j,area in enumerate(region_features)}
        # track region id
        region_idx = []
        for i, region in enumerate(all_sam):
            if region['region_id'] not in region_all.keys():
                continue
            else:
                region_idx.append(region_all[region['region_id']])
                region_order.append(region['region_id'])
                mask = mask_utils.decode(region['segmentation'])
                all_regions.append(mask.astype('float32'))

        if len(feature_all) == 0: # There were no predicted regions; use None as a flag
            all_pixel_predictions.append(None)
            continue

        features = torch.tensor(np.stack(feature_all))
        features = features[region_idx,:]

        predictions = torch.zeros((len(feature_all),args.num_classes+1))
        with torch.no_grad():
            feats = features

            model = model.cuda()

            feats = feats.cuda()

            output = model(feats.float())
            predictions = output.cpu()

        if 'after_softmax' in args.multi_region_pixels:
            # averaging softmax values for pixels in multiple regions
            class_predictions = softmax(predictions)
        else:
            # use logits for predictions
            class_predictions = predictions

        num_regions, num_classes = class_predictions.size()

        all_regions = torch.from_numpy(np.stack(all_regions,axis=-1))
        class_predictions = class_predictions.cuda()
        h,w,num_regions = all_regions.size()

        #find pixels where at least one mask equals one
        mask_sum = torch.sum(all_regions,dim=-1)

        mask_sum = mask_sum.cuda()
        nonzero_mask = torch.nonzero(mask_sum,as_tuple=True)

        all_regions = all_regions.cuda()

        nonzero_regions = all_regions[nonzero_mask[0],nonzero_mask[1],:]
        product = torch.matmul(nonzero_regions, class_predictions)

        # want avg across softmax values, need to get number of regions summed for each pixel
        # repeat number of regions across softmax values

        divide = torch.repeat_interleave(mask_sum[nonzero_mask[0],nonzero_mask[1],None],num_classes,dim=1)

        nonzero_region_pixel_preds = torch.divide(product,divide)

        if 'before_softmax' in args.multi_region_pixels:
            nonzero_region_pixel_preds = softmax(nonzero_region_pixel_preds,dim=1)
        top_pred = torch.argmax(nonzero_region_pixel_preds,dim=1).cpu().numpy()
        final_pixel_pred = np.zeros((h,w))

        # index back into original shape
        final_pixel_pred[nonzero_mask[0].cpu().numpy(),nonzero_mask[1].cpu().numpy()] = top_pred

        all_pixel_predictions.append(final_pixel_pred)
    return all_pixel_predictions, file_names
def train_and_evaluate(args):
    if args.num_classes != 150 and args.num_classes!= 20:
        raise ValueError('ADE should have 150 and Pascal VOC should have 20. The background class is taken care of in the code')
    if args.num_classes == 150:
        if args.ade ==False:
            raise ValueError('If using ADE then ade argument should be set to True')
    if args.ade==True:
        logger.info('Training and evaluating on ADE.')
    if not args.eval_only:
        train_model(args)

    all_pixel_predictions, file_names = eval_model(args)

    # Save pixel predictions as PNGs for use on evaluation server for Pascal VOC
    if args.output_predictions:
        logger.info('Saving predictions to PNGs')
        prediction_dir = os.path.join(args.save_dir, 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)

        for file_name, prediction in tqdm(zip(file_names, all_pixel_predictions)):
            prediction = Image.fromarray(prediction.astype(np.uint8))
            prediction.save(os.path.join(prediction_dir, file_name.replace('.pkl', '.png')))

    if not args.no_evaluation: # No need to output predictions if evaluating here
        compute_iou(args,all_pixel_predictions,file_names,args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_region_labels_dir",
        type=str,
        default=None,
        help="Location where ground truth label regions are stored for training images",
    )
    parser.add_argument(
        "--val_region_labels_dir",
        type=str,
        default=None,
        help="Location where ground truth label regions are stored for val images",
    )
    parser.add_argument('--epochs',
                        type=int,
                         default=2,
                         help='Number of iterations to run log regression')
    parser.add_argument(
        "--ignore_zero",
        action="store_true",
        help="Include 0 class"
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default=None,
        help="Location of region data."
    )
    parser.add_argument(
        "--val_data_file",
        type=str,
        default=None,
        help="Location of region data. Created if None"
    )
    parser.add_argument(
        "--train_region_feature_dir",
        type=str,
        default=None,
        help="Location of features for each region in training images"
    )
    parser.add_argument(
        "--val_region_feature_dir",
        type=str,
        default=None,
        help="Location of features for each region in val images"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Location to store trained classifiers"
    )

    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="No classifier training"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=.0001,
        help="learning rate"
    )

    parser.add_argument(
        "--sam_dir",
        type=str,
        default=None,
        help="SAM masks for eval"
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
        help="Location of ground truth annotations"
    )
    parser.add_argument(
        "--multi_region_pixels",
        type=str,
        default="avg_after_softmax",
        help="What to do for pixels in multiple regions. Default is average over probabilities after softmax"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        help="linear or mlp")
   
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="Number of classes in dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8192,
        help="Batch"
    )
    parser.add_argument(
        "--iou_every",
        type=int,
        default=1,
        help="Compute iou every"
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=512,
        help="hidden channel size if used"
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=1024,
        help="input channel size depending on models"
    )
    parser.add_argument(
        '--output_predictions',
        action='store_true',
        help='Output predictions as PNGs'
    )
    parser.add_argument(
        '--use_scheduler',
        action='store_true',
        help='Whether to use scheduler'
    )
    parser.add_argument(
        '--ade',
        action='store_true',
        help='Whether the datset we\'re running on is ADE20K. Adjusts labeling and loss computation'
    )

    parser.add_argument(
        '--override_ade_detection',
        action='store_true',
        help='Whether to train/eval anyways in spite of the ADE20K dataset detection.'
    )

    parser.add_argument(
        '--no_evaluation',
        action='store_true',
        help='Whether to skip evaluation (e.g. for Pascal VOC test which hasn\'t released labels.'
    )

    parser.add_argument(
        '--log_to_wandb',
        action='store_true',
        help='Whether to log results to wandb'
    )

    args = parser.parse_args()

    # Try to detect whether the dataset is ADE20K, and if so, force the user to set the flag
    dirs = [
        s.lower() for s in [
            args.train_region_labels_dir,
            args.val_region_labels_dir,
            args.annotation_dir,
            args.train_region_feature_dir,
            args.val_region_feature_dir,
            args.sam_dir
        ]
    ]

    if 'ade20k' in dirs and not args.override_ade_detection:
        raise ValueError('Detected ADE20K dataset. Please set the --ade flag.')

    # Save arguments to save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    if args.log_to_wandb:
        import wandb
        wandb.init(project='regions', config=args)

    train_and_evaluate(args)