import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import scipy.stats
import skimage
import logging
import random
from PIL import Image
from skimage import io
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    """ Basic Dataset for W-RIZZ """
    def __init__(self,
                 folder_path,
                 csv_path,
                 resolution, # should be (H, W)
                 augmentation=None,
                 consistency_augmentation=None, # if not none, we assume a mean-teacher style setup
                 in_memory=False,
                 normalize=True,
                 rebalancing='none'):
        """ Constructor for BasicDataset
        
        :param folder_path: path to folder containing data (which entries in csv_path are relative to)
        :param csv_path: path to data csv file
        :param resolution: resolution for images (H, W)
        :param augmentation: augmentation to use
        :param consistency_augmentation: mean teacher consistency augmentation to use
        :param in_memory: set to True if all samples should be loaded and kept in memory
        :param normalize: if True, images are normalized
        :param rebalancing: whether to rebalance data by oversampling minority class ('none' or 'default')
        """
        
        self._samples = None
        self._folder_path = Path(folder_path)
        self._csv = self._rebalance(pd.read_csv(csv_path), rebalancing)
        self._resolution = resolution # (H,W)
        self._augmentation = augmentation
        self._consistency_augmentation = consistency_augmentation
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]) if normalize else None
        
        if self._consistency_augmentation is not None:
            logging.info("Using mean teacher-style setup")
        else:
            logging.info("Using default setup")
        
        # If in_memory, let's read all data
        if in_memory:
            self._samples = [self._read_item(i) for i in range(self._csv.shape[0])]
            
    def _rebalance(self, csv, mode):
        """ Rebalance the data csv
        
        :param csv: csv to rebalance
        :param mode: rebalance mode
        :returns: balanced csv
        """
        # Check for valid rebalancing mode
        if mode not in ['none', 'default']:
            raise ValueError('Invalid rebalance mode provided!')
        
        # Convert from {-1,0,1} to {0,1} (where 0 = equality, 1 = inequality)
        get_ordinal = lambda x: 0 if int(x) == 0 else 1

        if mode != 'none':
            # Rebalance the dataset
            labels = np.array([scipy.stats.mode([get_ordinal(l[1:-1].split(',')[-1]) for l in csv.iloc[i]['labels'].split(';')], keepdims=False).mode.item() for i in range(csv.shape[0])])
            logging.info("Initial label statistics: {} (eq) - {} (gt/lt)".format((labels == 0).sum(), (labels == 1).sum()))
            num0, num1 = (labels==0).sum(), (labels==1).sum()
            if num0 > num1:
                excess = num0 - num1
                selection_csv = pd.concat([csv[labels == 1].reset_index(drop=True)]*math.ceil(excess/num1), ignore_index=True)
                csv = pd.concat([csv, selection_csv.iloc[random.sample(list(range(selection_csv.shape[0])), k=excess)]]).reset_index(drop=True)
            elif num0 < num1:
                excess = num1 - num0
                selection_csv = pd.concat([csv[labels == 0].reset_index(drop=True)]*math.ceil(excess/num0), ignore_index=True)
                csv = pd.concat([csv, selection_csv.iloc[random.sample(list(range(selection_csv.shape[0])), k=excess)]]).reset_index(drop=True)
        
        # Count the new labels
        new_labels = np.array([scipy.stats.mode([get_ordinal(l[1:-1].split(',')[-1]) for l in csv.iloc[i]['labels'].split(';')], keepdims=False).mode.item() for i in range(csv.shape[0])])
        logging.info("Final label statistics: {} (eq) - {} (gt/lt)".format((new_labels == 0).sum(), (new_labels == 1).sum()))
        
        return csv
    
    def _read_item(self, idx):
        """ Read a data item
        
        :param idx: index in dataset of item to read
        :returns: (ImageTensor, ImageTensor, AnnoTensor)
        """
        
        entry = self._csv.iloc[idx]
        imageA_file = self._folder_path / entry['imageA_name']
        imageB_file = self._folder_path / entry['imageB_name']
        width, height = entry['width'], entry['height']
        annotation_str = entry['labels']
        
        # Read image
        imageA = io.imread(imageA_file)[:,:,:3] # :3 to remove potential alpha channel
        imageA = Image.fromarray(skimage.img_as_ubyte(self._rescale_color(imageA, resolution=self._resolution)))
        imageB = io.imread(imageB_file)[:,:,:3] # :3 to remove potential alpha channel
        imageB = Image.fromarray(skimage.img_as_ubyte(self._rescale_color(imageB, resolution=self._resolution)))
        
        # Read annotations
        annotations = []
        for a in annotation_str.split(';'):
            l = [int(s) for s in a[1:-1].split(',')]
            scale_x = self._resolution[1] / width
            scale_y = self._resolution[0] / height
            # l[-1] == 0 => eq; l[-1] == 1 => latter is more; l[-1] == -1 => former is more
            annotations.append((
                # ensure that annotations are properly resized to desired resolution
                l[0], # 0 if pt1 is in imgA, 1 if it's in imgB
                max(0, min(round(scale_x * l[1]), self._resolution[1]-1)),
                max(0, min(round(scale_y * l[2]), self._resolution[0]-1)),
                l[3], # 0 if pt2 is in imgA, 1 if it's in imgB
                max(0, min(round(scale_x * l[4]), self._resolution[1]-1)),
                max(0, min(round(scale_y * l[5]), self._resolution[0]-1)),
                l[6]
            ))
        return (transforms.ToTensor()(imageA), transforms.ToTensor()(imageB), torch.tensor(annotations, dtype=torch.long))
        
    def _rescale_color(self, image, resolution):
        """ Rescale a color image
        
        :param image: image to rescale
        :param resolution: desired resolution in form (H, W) (if None, image will be returned)
        :returns: rescaled image
        """
        if resolution is None:
            return image
        else:
            rescaled = skimage.transform.resize(image, resolution)
            return rescaled
    
    def __len__(self):
        """ Returns size of dataset
        
        Note: each entry technically contains 2 images (due to cross-image labeling)
        """
        return self._csv.shape[0]
    
    def __getitem__(self, idx, augment=True, normalize=True):
        """" Gets an item
        
        Note: if mean teacher is used, we return 4 images (two views for each image). If not, we only
        return 2 images.
        
        :param idx: index to fetch
        :param augment: whether to apply augmentations
        :param normalize: whether to normalize images
        :returns: (ImageTensor, ImageTensor, AnnoTensor) if self._consistency_augmentation is None, or
                  (ImageTensor, ImageTensor, ImageTensor, ImageTensor, AnnoTensor) otherwise
        """
        if self._samples is None: # if not in memory already, read
            item = self._read_item(idx)
        else:
            item = self._samples[idx]
        
        # apply augmentation
        if self._augmentation is not None and augment:
            item = self._augmentation(item)
        
        # apply consistency augmentation (for mean teacher)
        if self._consistency_augmentation is not None:
            item = (item[0], item[1], item[0], item[1], item[2])
            if augment:
                item = (*[self._consistency_augmentation(d) for d in item[0:-1]], item[-1])
        
        # normalize RGB images
        if self._normalize is not None and normalize:
            item = (*[self._normalize(d) for d in item[0:-1]], item[-1])
        
        return item
