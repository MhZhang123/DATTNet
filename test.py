import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import cv2
from torchvision import transforms
from PIL import Image

from modules import *
from dataset import *
from loss import *
from utils import *

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--bt_size', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--deterministic', type=int,  default=0,
                    help='whether use deterministic training')

    
    parser.add_argument('--cfg', default='a', type=str, metavar="FILE", help='path to config file', )
    
    parser.add_argument('--seed', type=int, default=1232)
    parser.add_argument('--batch_size', type=int,
                    default=48, help='batch_size per gpu')
    
    parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
    parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')    
    parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+')    
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')   
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
    
    #################test parameter#########################  
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
    
    parser.add_argument('--use_kpg', type=bool, default=True)
    
    
    args = parser.parse_args(args=[])
    return args
args = get_cfg()


x_transforms = transforms.Compose([transforms.ToTensor()])
y_transforms = transforms.ToTensor()
batch_size = args.batch_size

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


#################dataset#########################
dataset_config = {
    'Kvasir-SEG':{'Dataset': mydataset,
          'volume_path': './dataset/Kvasir-SEG/test',
          'root_path': './dataset/Kvasir-SEG/train',
          'list_dir':'./dataset/Kvasir-SEG',
          'num_class': 2,
          'z_spacing': 1},       
    
}
###############init#####################
args.dataset = 'Kvasir-SEG'  # ACDC, Synapse, Kvasir-SEG

dataset_name = args.dataset
args.num_classes = dataset_config[dataset_name]['num_class']
args.volume_path = dataset_config[dataset_name]['volume_path']
args.root_path = dataset_config[dataset_name]['root_path']
args.Dataset = dataset_config[dataset_name]['Dataset']
args.list_dir = dataset_config[dataset_name]['list_dir']
args.z_spacing = dataset_config[dataset_name]['z_spacing']
args.num_class = dataset_config[dataset_name]['num_class']


if dataset_name == 'Kvasir-SEG':
    db_test = mydataset_poly(split='test', aug=True)
    testloader = torch.utils.data.DataLoader(db_test, batch_size=16, shuffle=False, num_workers=4, pin_memory=False)

else:
    db_test = args.Dataset(num_class=args.num_class, base_dir=args.volume_path, list_dir=args.list_dir, split="test_vol",img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    
def inference(epoch, args, model, case, test_save_path):
    global best_performance
    global save_mode_path
    global iter_num
    metric_list = 0.0
    model.eval()
    with torch.no_grad():
        if args.dataset == 'Kvasir-SEG':
            metrics = []
            all_case = []
            for i, sampled_batch in enumerate(testloader):
                image, label = sampled_batch["image"], sampled_batch["label"]
                case = sampled_batch["case_name"]
                label = label.squeeze(1).cpu().detach().numpy()
                image = image.cuda()
                model.eval()
                with torch.no_grad():
                    if args.model=='Ours':
                        seg, ps = model(image)
                        out = torch.argmax(torch.softmax(seg, dim=1), dim=1).squeeze(0)
                    else:
                        out = torch.argmax(torch.softmax(model(image), dim=1), dim=1).squeeze(0)
                    prediction = out.cpu().detach().numpy().astype(np.float32)
                pred = out.cpu().detach().numpy()
                image = image.cpu().detach().numpy()
                metrics.append(calculate_metric_percase(prediction == 1, label == 1))
            metrics = np.array(metrics)    
            performance, mean_hd95= np.mean(metrics, axis=0)          
            return performance, mean_hd95, ps_0
        else:
            for i_batch, sampled_batch in enumerate(testloader):
                h, w = sampled_batch["image"].size()[2:]
                image, label = sampled_batch["image"], sampled_batch["label"]
                metric_i = test_single_volume_visualization(image, label, model, classes=args.num_classes, patch_size=[args.img_size,args.img_size],modelname=args.model, volume_path=path, case=case, z_spacing=args.z_spacing)
                metric_list += np.array(metric_i[0])  
            metric_list = metric_list / len(db_test) 
            performance, mean_hd95 = np.mean(metric_list, axis=0)
        
    return performance, mean_hd95, metric_list, metric_i[-3], metric_i[-2], metric_i[-1]
    
    
    
    
