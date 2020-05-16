import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import random
import glob

class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir, mode, 'img')
        self.seg_dir = os.path.join(self.data_dir, mode, 'seg')

        img_list, seg_list = [], []
        for path in glob.glob(self.img_dir + '/*.png'):
            img_list.append(path)
        for path in glob.glob(self.seg_dir + '/*.png'):
            seg_list.append(path)
        img_list.sort()
        seg_list.sort()
        self.img = img_list
        self.seg = seg_list

    def __len__(self):
        return len(self.img)

    def transform(self, image, mask):
        if self.mode == 'train':
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(352, 448))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.img[idx]
        seg_path = self.seg[idx]
        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(seg_path)
        x, y = self.transform(img, mask)

        if self.mode == 'val':
            return x, y, img_path
        else:
            return x, y


class TESTDATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.img_dir = args.img_dir

        img_list = []
        for path in glob.glob(self.img_dir + '/*.png'):
            img_list.append(path)
        img_list.sort()
        self.img = img_list

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.img[idx]
        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        image = TF.to_tensor(img)
        image = TF.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        return image, img_path
