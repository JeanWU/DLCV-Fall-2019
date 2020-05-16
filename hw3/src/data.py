import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import csv
import torch
import glob

class DATA(Dataset):
    def __init__(self, args, mode=None, dataset=None):

        ''' set up basic parameters for dataset '''
        self.data_dir = args.data_dir
        self.transform = transforms.Compose([
                           #transforms.RandomHorizontalFlip(0.5),
                           transforms.Resize(args.image_size),
                           transforms.CenterCrop(args.image_size),
                           transforms.ToTensor(),   # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

        img_list = []
        self.mode = mode
        if args.model_type in ('gan', 'acgan'):
            self.img_dir = os.path.join(self.data_dir, 'face', 'train')
            self.csv_file = os.path.join(self.data_dir, 'face', 'train.csv')
            #self.csv_file = '../train.csv'
            with open(self.csv_file, newline='') as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    img_list.append((row['image_name'], row['Smiling']))

        if args.model_type in ('dann'):
            self.img_dir = os.path.join(self.data_dir, 'digits', dataset, mode)
            self.csv_file = os.path.join(self.data_dir, 'digits', dataset, mode+'.csv')
            with open(self.csv_file, newline='') as csvfile:
                rows = csv.DictReader(csvfile)
                for row in rows:
                    img_list.append((row['image_name'], row['label']))


        self.img = img_list


    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):

        ''' get data '''
        image_name, classes = self.img[idx]
        img_path = os.path.join(self.img_dir, image_name)

        ''' read image '''
        img = Image.open(img_path).convert('RGB')

        #return self.transform(img), torch.LongTensor([float(classes)])
        """
        if self.mode == 'train':
            return self.transform(img), torch.tensor(float(classes))
        elif self.mode == 'test':
            return self.transform(img), torch.tensor(float(classes)), image_name
        """
        return self.transform(img), torch.tensor(float(classes))




class TESTDATA(Dataset):
    def __init__(self, args):

        ''' set up basic parameters for dataset '''
        self.img_dir = args.data_dir
        self.transform = transforms.Compose([
                           #transforms.RandomHorizontalFlip(0.5),
                           transforms.Resize(args.image_size),
                           transforms.CenterCrop(args.image_size),
                           transforms.ToTensor(),   # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

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
        img_name = img_path[-9:]

        return self.transform(img), img_name
