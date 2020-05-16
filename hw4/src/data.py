import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import reader
import os
import glob
from PIL import Image


class DATA(Dataset):
    def __init__(self, args, mode=None):

        ''' set up basic parameters for dataset '''
        self.transform = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.Resize((224, 224)),
                           transforms.ToTensor(),   # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
        self.rescale_factor = args.rescale_factor
        if mode == 'train':
            self.video_path = os.path.join(args.video_path, 'train')
            csv_path = os.path.join(args.csv_path, 'gt_train.csv')
        elif mode == 'test':
            self.video_path = os.path.join(args.video_path, 'valid')
            csv_path = os.path.join(args.csv_path, 'gt_valid.csv')
        else:
            self.video_path = args.video_path
            csv_path = args.csv_path

        video_dict = reader.getVideoList(csv_path)
        self.video_category_list = video_dict['Video_category']
        self.video_name_list = video_dict['Video_name']
        self.label_list = video_dict['Action_labels']


    def __len__(self):
        return len(self.video_name_list)

    def __getitem__(self, idx):

        ''' get data '''
        video_category = self.video_category_list[idx]
        video_name = self.video_name_list[idx]
        label = self.label_list[idx]
        frames = reader.readShortVideo(self.video_path, video_category, video_name, downsample_factor=12, rescale_factor=self.rescale_factor)
        #print("frames.shape: ",frames.shape)


        frames_list = []
        for f in range(frames.shape[0]):
            frame = frames[f,:,:,:]
            frame = self.transform(frame)
            frames_list.append(frame)

        """
        #handle every sample, then concat 
        imgs = []
        for frame in frames:
            imgs.append(self.transform(frame))
        imgs = np.array(imgs)
        print("imgs shape ",imgs.shape)
        """


        return torch.stack(frames_list), torch.tensor(int(label))
