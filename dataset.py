from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import imgaug as ia
import imgaug.augmenters as iaa  # 导入iaa
from torchvision import transforms
import os
import random
import h5py
import numpy as np
import torch
import albumentations as A

import os
import glob
import json
import torch
import random
import torch.nn as nn
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import albumentations as A

def mask_to_onehot(mask, num_class):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask,-1)
    for colour in range (num_class):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map

def augment_seg(img_aug, img, seg, num_class):
    seg = mask_to_onehot(seg, num_class)
    aug_det = img_aug.to_deterministic() 
    image_aug = aug_det.augment_image( img )

    segmap = ia.SegmentationMapOnImage( seg , nb_classes=np.max(seg)+1 , shape=img.shape )
    segmap_aug = aug_det.augment_segmentation_maps( segmap )
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug , segmap_aug

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
    
    
class mydataset(Dataset):
    def __init__(self, base_dir, list_dir, split, img_size, num_class, norm_x_transform=None, norm_y_transform=None):
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.img_size = img_size
        self.num_class = num_class
        
        self.img_aug = iaa.SomeOf((0,4),[
            iaa.Flipud(0.5, name="Flipud"),  
            iaa.Fliplr(0.5, name="Fliplr"),  
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),  
            iaa.GaussianBlur(sigma=(1.0)),  
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5), 
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),  
            iaa.Affine(rotate=(-40, 40)), 
            iaa.Affine(shear=(-16, 16)),  
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),  
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)
        

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
   

            image,label = augment_seg(self.img_aug, image, label, self.num_class)

            x, y = image.shape
            if x != self.img_size or y != self.img_size:
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3)  # why not 3?
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)
                

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}

        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'])

        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'])
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
    
    
    
def norm01(x):
    return np.clip(x, 0, 255) / 255


class mydataset_poly(data.Dataset):
    def __init__(self, split, size=224, aug=False, polar=False):
        super(mydataset_poly, self).__init__()
        self.polar = polar 
        self.split = split 

        self.image_paths = []
        self.label_paths = []
        self.dist_paths = []

        root_dir = './dataset/Kvasir-SEG'

        if split == 'train':
            indexes = os.listdir(root_dir + "/train/Image")
            self.image_paths = [f'{root_dir}/train/Image/{_id}' for _id in indexes]
            self.label_paths = [f'{root_dir}/train/Label/{_id}' for _id in indexes]
        else:
            indexes = os.listdir(root_dir + "/test/Image")
            self.image_paths = [f'{root_dir}/test/Image/{_id}' for _id in indexes]
            self.label_paths = [f'{root_dir}/test/Label/{_id}' for _id in indexes]
            
        self.num_samples = len(self.image_paths)
        self.aug = aug  # 1
        self.size = size
        p = 0.5
        self.transf = A.Compose([A.GaussNoise(p=p), A.HorizontalFlip(p=p), A.VerticalFlip(p=p),A.ShiftScaleRotate(p=p)])
        
    def __getitem__(self, index):
        image_data = np.load(self.image_paths[index])
        case_name = os.path.basename(self.image_paths[index]).split('.')[0]
    
        image_data = np.array(cv2.resize(image_data, (self.size, self.size), cv2.INTER_LINEAR))
   
        
        label_data = (np.load(self.label_paths[index])).astype('uint8')
        
        label_data = np.array(cv2.resize(label_data, (self.size, self.size), cv2.INTER_NEAREST))


        if self.aug and self.split == 'train':
            mask = label_data
            
            tsf = self.transf(image=image_data.astype('uint8'), mask=mask)  
          
            image_data, mask_aug = tsf['image'], tsf['mask']           
            
            label_data = mask_aug

        image_data = norm01(image_data) 
        label_data = np.expand_dims(label_data, 0)

        image_data = torch.from_numpy(image_data).float()
        label_data = label_data / 255.0 > 0.5
        label_data = torch.from_numpy(label_data).float()
        
     
        image_data = image_data.permute(2, 0, 1)
        
        return {
            'image': image_data,
            'label': label_data,
            'case_name':case_name
        }

    def __len__(self):
        return self.num_samples
    
