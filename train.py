import os, argparse, math
import numpy as np
from glob import glob
from tqdm import tqdm
import sys
import time
import logging
import shutil
from medpy import metric

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from medpy.metric.binary import hd, dc, assd, jc
from sklearn.metrics import confusion_matrix,cohen_kappa_score
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

import torch, gc
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

import random
import matplotlib.pyplot as plt
from modules import *
from loss import *
from utils import *
import warnings
warnings.filterwarnings("ignore")


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
    
    parser.add_argument('--cfg', default='a', type=str, metavar="FILE", help='path to config file')
    # parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
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


args.dataset = 'Kvasir-SEG'  # ACDC, Synapse, Kvasir-SEG 

args.batch_size = 8

if args.dataset == 'Synapse':
    args.n_epochs = 150
elif args.dataset == 'ACDC':
    args.n_epochs = 200

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)   
random.seed(args.seed)
np.random.seed(args.seed)


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# !set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
if not args.deterministic:
    print(f'1')
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    print(f'2')
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    
if args.dataset == 'Kvasir-SEG':
    x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
else:
    x_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
y_transforms = transforms.ToTensor()
batch_size = args.batch_size

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def compute_acc(gt, pred):
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(pred).flatten())
    acc = np.diag(matrix).sum() / matrix.sum()
    return acc

def compute_iou(input, target, classes=1):
    """  compute the value of iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        iou: float, the value of iou
    """
    input = np.array(input)
    target = np.array(target)
    intersection = np.logical_and(target == classes, input == classes)
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error，MAE

    param：
    y_true: array-like
    y_pred: array-like

    return：
    mae
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    absolute_errors = np.abs(y_pred-y_true)
    mae = np.mean(absolute_errors)
    
    return mae        
    
def calculate_metric_percase_kvasir(pred, gt):  
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:  
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        mae = mean_absolute_error(gt, pred)
        acc = compute_acc(gt, pred)
        iou = compute_iou(pred, gt, classes=1)
        
        return dice, hd95, mae, acc, iou  
    elif pred.sum() > 0 and gt.sum()==0:  
        
        return 1, 0
    else:
        return 0, 0  

from dataset import *
dataset_config = {
    'ACDC': {'Dataset': mydataset,
             'volume_path': '/root/autodl-tmp/dataset/ACDC/test_vol_h5',
             'root_path': '/root/autodl-tmp/dataset/ACDC/train_npz',
             'list_dir': '/root/autodl-tmp/dataset/ACDC',
             'num_class': 4,
             'z_spacing': 1,},
    'Synapse':{'Dataset': mydataset,
              'volume_path': '/root/autodl-tmp/project_TransUNet/data/Synapse/test_vol_h5',
              'root_path': '/root/autodl-tmp/project_TransUNet/data/Synapse/train_npz',
              'list_dir':'/root/autodl-tmp/MISSFormer-main/lists/lists_Synapse',
              'num_class': 9,
              'z_spacing': 1,},
    'Kvasir-SEG':{'Dataset': mydataset,
          'volume_path': '/root/autodl-tmp/dataset/Kvasir-SEG/test_vol_h5',
          'root_path': '/root/autodl-tmp/dataset/Kvasir-SEG/train_npz',
          'list_dir':'/root/autodl-tmp/dataset/Kvasir-SEG',
          'num_class': 2,
          'z_spacing': 1,}
}


dataset_name = args.dataset
args.num_classes = dataset_config[dataset_name]['num_class']
args.volume_path = dataset_config[dataset_name]['volume_path']
args.root_path = dataset_config[dataset_name]['root_path']
args.Dataset = dataset_config[dataset_name]['Dataset']
args.list_dir = dataset_config[dataset_name]['list_dir']
args.z_spacing = dataset_config[dataset_name]['z_spacing']
args.num_class = dataset_config[dataset_name]['num_class']

if dataset_name == 'Kvasir-SEG':
    db_train = mydataset_poly(split='train', aug=True)
    trainloader = torch.utils.data.DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, worker_init_fn=worker_init_fn)

    db_test = mydataset_poly(split='test', aug=True)
    testloader = torch.utils.data.DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False, worker_init_fn=worker_init_fn)

else:
    db_train = mydataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",img_size=args.img_size, num_class = args.num_classes,norm_x_transform = x_transforms, norm_y_transform = y_transforms)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    db_test = args.Dataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test_vol",img_size=args.img_size, num_class = args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1, worker_init_fn=worker_init_fn)
    
from modules import DATTNet
model = DATTNet(out_channels=args.num_classes)   
model = model.to(args.device)



# from thop import profile, clever_format
# input = torch.randn(1, 3, 224, 224).cuda()
# flops, params = profile(model, inputs=(input, ))
# flops, params = clever_format([flops, params], "%.3f")
# print('params:', params)
# print('flops:', flops)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
    

def validation(epoch, args, model, test_save_path=None):
    global best_performance
    global save_mode_path
    global iter_num
    global save_dir_path
    metric_list = 0.0
    model.eval()
    with torch.no_grad():
        
        if args.dataset == 'Kvasir-SEG':
            metrics = []
            for sampled_batch in testloader:
                image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
                label = label.squeeze(1).cpu().detach().numpy()
                image = image.cuda()
                model.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(model(image), dim=1), dim=1).squeeze(0)
                    prediction = out.cpu().detach().numpy().astype(np.float32)

                metrics.append(calculate_metric_percase(prediction == 1, label == 1))

            metrics = np.array(metrics)    
            performance, mean_hd95= np.mean(metrics, axis=0)          
        
        else:
            for i_batch, sampled_batch in enumerate(testloader):
                h, w = sampled_batch["image"].size()[2:]

                image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

                metric_i, image, label, prediction = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                              test_save_path=None, case=case_name, z_spacing=args.z_spacing)

                metric_list += np.array(metric_i) 

            metric_list= metric_list / len(db_test)
            
            performance, mean_hd95 = np.mean(metric_list, axis=0)
        
        writer.add_scalar('performance', performance.round(4), epoch)
        
        
        if performance > best_performance:
            best_performance = performance
            if save_mode_path is not None:
                os.remove(save_mode_path)

            writer.add_scalar('best_performance', best_performance.round(4), epoch)
            t = time.strftime("%Y-%m-%d-%X", time.localtime())
            save_name = f"{t}_{args.model}_{args.dataset}_epo{str(epoch)}_perf{best_performance:.3f}_lr{args.lr}.pth"
            save_mode_path = os.path.join(args.saved_mode_weight, save_name)
            torch.save(model.state_dict(), save_mode_path)
            
        print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return prediction, label, image



optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=3e-5, amsgrad=False)

def create_lr_scheduler(optimizer,
                        num_step,
                        epoch,
                        warmup=True,
                        warmup_epoch=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epoch > 0
    
    if warmup is False:
        warmup_epoch = 0
        
    def f(x):
        if warmup is True and x < (warmup_epoch * num_step):
            alpha = float(x) / (epoch * num_step)
            return alpha + (1 - alpha) * warmup_factor
        else:
            return (1 - (x - warmup_epoch * num_step) / ((epoch - warmup_epoch) * num_step)) ** 0.9
        
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

lr_scheduler = create_lr_scheduler(optimizer, len(trainloader), args.n_epochs)

def criterion_v1(outputs, label_batch):
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_class)
    loss_ce = ce_loss(outputs, label_batch[:].long())
    loss_dice = dice_loss(outputs, label_batch, softmax=True)
    loss = 0.4 * loss_ce + 0.6 * loss_dice 
    return loss

# log
logger = logging.getLogger(f'dataset:{args.dataset}_optimizer:{args.optimizer}_model:{args.model}_batchsize:{args.batch_size}_lr{args.lr}')  
handler = logging.FileHandler("/root/autodl-tmp/________new__________/log/log.txt", mode='a', encoding='utf-8', delay=False)  
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%B-%d %A %H:%M:%S') 
handler.setFormatter(formatter)  
logger.addHandler(handler)  
logger.setLevel(logging.DEBUG)  

best_performance = 0
save_mode_path = None
save_dir_path = None
iter_num = 0
max_iterations = args.n_epochs * len(trainloader)


for epoch in tqdm(range(1, args.n_epochs+1)):
    model.train()
    metric_list = 0.0
    a = time.time()
    for i, batch in enumerate(trainloader): 

        optimizer.zero_grad()
        image_batch, label_batch, case_name= batch['image'], batch['label'], batch['case_name'][0]
        if i == 0:

            train_image_example = image_batch[4, ...].numpy()
            max_img = train_image_example.max()
            min_img = train_image_example.min()
            train_image_example = 225 * ((train_image_example - min_img) / (max_img - min_img))
            train_image_example = train_image_example.astype(np.uint8)
            train_label_example = label_batch[4, ...].numpy()
            max_label = train_label_example.max()
            min_label = train_label_example.min()
            train_label_example = 225 * (train_label_example / (max_label - min_label))
            train_label_example = train_label_example.astype(np.uint8)
            
            
        
        image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()  # [8, 1, 224, 224]
        outputs = model(image_batch)

        loss = criterion_v1(outputs, label_batch)
        loss.backward()
        optimizer.step()
        
        lr_ = args.lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        iter_num = iter_num + 1

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    s = time.time()
    prediction, label, image = validation(epoch, args, model, test_save_path)        