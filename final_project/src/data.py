import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms.functional as TF
import random
import glob
import cv2


class DATA(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        ''' set up basic parameters for dataset '''
        img_dir = os.path.join(args.data_dir, mode)
        gt_dir = os.path.join(args.data_dir, mode+'_gt')
        """
        img_list, mask_list, gt_list = [], [], []
        for path in glob.glob(img_dir + '/*_mask.jpg'):
            mask_list.append(path)
            img_num = path[len(img_dir)+1:-9]
        
        for path in glob.glob(img_dir + '/*_masked.jpg'):
            img_list.append(path)
        for path in glob.glob(gt_dir + '/*.jpg'):
            gt_list.append(path)
            
        img_list.sort()
        mask_list.sort()
        gt_list.sort()
        """
        num_list = []
        for path in glob.glob(img_dir + '/*_mask.jpg'):
            img_num = path[len(img_dir):-9]
            num_list.append(img_num)

        num_list.sort()
        img_list, mask_list, gt_list = [], [], []
        for i in num_list:
            img_list.append(img_dir + '/{}_masked.jpg'.format(i))
            mask_list.append(img_dir + '/{}_mask.jpg'.format(i))
            gt_list.append(gt_dir + '/{}.jpg'.format(i))

        self.img = img_list
        self.mask = mask_list
        self.gt = gt_list

    def __len__(self):
        return len(self.img)

    def transform(self, image, mask, gt):
        """ pad image size to multiple of 128 """
        h, w = image.size
        h_mul = int(h/128) + 1
        w_mul = int(w/128) + 1
        new_h = h_mul * 128
        new_w = w_mul * 128
        delta_h = new_h - h
        delta_w = new_w - w
        padding = (delta_h//2, delta_w//2, delta_h-(delta_h//2), delta_w-(delta_w//2))
        image = ImageOps.expand(image, padding)
        mask = ImageOps.expand(mask, padding)
        gt = ImageOps.expand(gt, padding)
        crop_box = (delta_h//2, delta_w//2, h+delta_h//2, w+delta_w//2)

        if self.mode == 'train':
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                gt = TF.hflip(gt)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                gt = TF.vflip(gt)

        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        gt = TF.to_tensor(gt)
        gt = TF.normalize(gt, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # new_mask = Image.eval(mask, lambda px: 0 if px == 0 else 1)
        # new_mask = torch.from_numpy(np.array(new_mask)).float()
        # new_mask = TF.to_tensor(new_mask)
        mask = TF.to_tensor(mask)
        return image, mask, gt, crop_box

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.img[idx]
        mask_path = self.mask[idx]
        gt_path = self.gt[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        img, mask, gt, crop_box = self.transform(img, mask, gt)

        return img, mask, gt, crop_box


class TEST_DATA(Dataset):
    def __init__(self, args):
        ''' set up basic parameters for dataset '''
        self.img_dir = args.test_dir
        img_list, mask_list = [], []
        for path in glob.glob(os.path.join(self.img_dir, '*_mask.jpg')):
            mask_list.append(path)

        for path in glob.glob(os.path.join(self.img_dir, '*_masked.jpg')):
            img_list.append(path)

        img_list.sort()
        mask_list.sort()

        self.img = img_list
        self.mask = mask_list

    def __len__(self):
        return len(self.img)

    def transform(self, image, mask):
        """ pad image size to multiple of 128 """
        h, w = image.size
        h_mul = int(h/128) + 1
        w_mul = int(w/128) + 1
        new_h = h_mul * 128
        new_w = w_mul * 128
        delta_h = new_h - h
        delta_w = new_w - w
        padding = (delta_h//2, delta_w//2, delta_h-(delta_h//2), delta_w-(delta_w//2))
        image = ImageOps.expand(image, padding)
        mask = ImageOps.expand(mask, padding)
        crop_box = (delta_h//2, delta_w//2, h+delta_h//2, w+delta_w//2)

        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask = TF.to_tensor(mask)
        return image, mask, crop_box

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.img[idx]
        mask_path = self.mask[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        img, mask, crop_box = self.transform(img, mask)

        return img, mask, crop_box
