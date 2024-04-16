import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random


class SparseHorizontalFlip:
    """ Augmentation for doing a horizonal flip """
    def __init__(self, p=0.5):
        """ Constructs augmentation
        
        :param p: probability for random flip
        """
        
        self._p = p
    
    def __call__(self, data):
        """ Apply augmentation
        
        :param data: data in the form of (ImageTensor, ImageTensor, AnnoTensor)
        """
        
        max_x = data[0].shape[-1] - 1
        annotation = data[2].clone()
        if torch.rand(1) < self._p:
            # Flip first image and annotation
            imageA = torchvision.transforms.functional.hflip(data[0])
            annotation[annotation[:,0] == 0,1] = max_x - annotation[annotation[:,0] == 0,1]
            annotation[annotation[:,3] == 0,4] = max_x - annotation[annotation[:,3] == 0,4]
        else:
            imageA = data[0]
        
        if torch.rand(1) < self._p:
            # Flip second image and annotation
            imageB = torchvision.transforms.functional.hflip(data[1])
            annotation[annotation[:,0] == 1,1] = max_x - annotation[annotation[:,0] == 1,1]
            annotation[annotation[:,3] == 1,4] = max_x - annotation[annotation[:,3] == 1,4]
        else:
            imageB = data[1]
        return (imageA, imageB, annotation)


class SparseRandomCrop:
    """ Augmentation for random cropping """
    def __init__(self, factor=0.25, p=0.5):
        """ Constructs augmentation
        
        :factor: factor by how much to crop on each axis
        :param p: probability for cropping
        """
        
        self._factor = factor
        self._p = p
    
    def __call__(self, data):
        """ Apply augmentation
        
        :param data: data in the form of (ImageTensor, ImageTensor, AnnoTensor)
        """
        
        w, h = data[0].shape[-1], data[0].shape[-2]
        annotation = data[2].clone()

        # we need to make sure that we don't crop out the annotation points
        hf = self._factor / 2
        min_x = min(annotation[:,1].min().cpu().item(), annotation[:,4].min().cpu().item())
        max_x = max(annotation[:,1].max().cpu().item(), annotation[:,4].max().cpu().item())
        min_y = min(annotation[:,2].min().cpu().item(), annotation[:,5].min().cpu().item())
        max_y = max(annotation[:,2].max().cpu().item(), annotation[:,5].max().cpu().item())

        if torch.rand(1) < self._p:
            # Get the bounds for cropping
            x_lower_A = max(min(random.randint(0, min(int(hf * w), min_x)), w-1), 0)
            x_upper_A = max(min(random.randint(max(int((1.0-hf) * w), max_x), w-1), w-1), 0)
            y_lower_A = max(min(random.randint(0, min(int(hf * h), min_y)), h-1), 0)
            y_upper_A = max(min(random.randint(max(int((1.0-hf) * h), max_y), h-1), h-1), 0)
            new_width_A = (x_upper_A - x_lower_A) + 1
            new_height_A = (y_upper_A - y_lower_A) + 1
            # Crop image
            imageA = torchvision.transforms.functional.resized_crop(data[0], y_lower_A, x_lower_A,
                                                                    new_height_A, new_width_A, size=(h, w),
                                                                    antialias=True)
            # Crop annotation
            annotation[annotation[:,0] == 0,1] = torch.clamp((annotation[annotation[:,0] == 0,1] - x_lower_A) * (w/new_width_A), min=0, max=w-1).to(dtype=data[2].dtype)
            annotation[annotation[:,3] == 0,4] = torch.clamp((annotation[annotation[:,3] == 0,4] - x_lower_A) * (w/new_width_A), min=0, max=w-1).to(dtype=data[2].dtype)
            annotation[annotation[:,0] == 0,2] = torch.clamp((annotation[annotation[:,0] == 0,2] - y_lower_A) * (h/new_height_A), min=0, max=h-1).to(dtype=data[2].dtype)
            annotation[annotation[:,3] == 0,5] = torch.clamp((annotation[annotation[:,3] == 0,5] - y_lower_A) * (h/new_height_A), min=0, max=h-1).to(dtype=data[2].dtype)
        else:
            imageA = data[0]
        
        if torch.rand(1) < self._p:
            # Get the bounds for cropping
            x_lower_B = max(min(random.randint(0, min(int(hf * w), min_x)), w-1), 0)
            x_upper_B = max(min(random.randint(max(int((1.0-hf) * w), max_x), w-1), w-1), 0)
            y_lower_B = max(min(random.randint(0, min(int(hf * h), min_y)), h-1), 0)
            y_upper_B = max(min(random.randint(max(int((1.0-hf) * h), max_y), h-1), h-1), 0)
            new_width_B = (x_upper_B - x_lower_B) + 1
            new_height_B = (y_upper_B - y_lower_B) + 1
            # Crop image
            imageB = torchvision.transforms.functional.resized_crop(data[1], y_lower_B, x_lower_B,
                                                                    new_height_B, new_width_B, size=(h, w),
                                                                    antialias=True)
            # Crop annotation
            annotation[annotation[:,0] == 1,1] = torch.clamp((annotation[annotation[:,0] == 1,1] - x_lower_B) * (w/new_width_B), min=0, max=w-1).to(dtype=data[2].dtype)
            annotation[annotation[:,3] == 1,4] = torch.clamp((annotation[annotation[:,3] == 1,4] - x_lower_B) * (w/new_width_B), min=0, max=w-1).to(dtype=data[2].dtype)
            annotation[annotation[:,0] == 1,2] = torch.clamp((annotation[annotation[:,0] == 1,2] - y_lower_B) * (h/new_height_B), min=0, max=h-1).to(dtype=data[2].dtype)
            annotation[annotation[:,3] == 1,5] = torch.clamp((annotation[annotation[:,3] == 1,5] - y_lower_B) * (h/new_height_B), min=0, max=h-1).to(dtype=data[2].dtype)
        else:
            imageB = data[1]

        return (imageA, imageB, annotation)


class ImageOnlyAugment:
    """ Augmentation that is only applied to images """
    def __init__(self, augmentation, p=0.5):
        """ Constructs augmentation
        
        :augmentation: augmentation to apply to image(s)
        :param p: probability of applying augmentation
        """
        self._augmentation = augmentation
        self._p = p
    
    def __call__(self, data):
        """ Apply augmentation
        
        :param data: data in the form of (ImageTensor, ..., ImageTensor, AnnoTensor)
        """
        if torch.rand(1) < self._p:
            return (*[self._augmentation(d) for d in data[0:-1]], data[-1])
        else:
            return data
