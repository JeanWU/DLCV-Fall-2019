#python hw4_p3.py ../hw4_data/FullLengthVideos/videos/valid ./output/
import data
import torch
import parser
import models
import os
import numpy as np
import utils
import torchvision.transforms as transforms
import glob
from PIL import Image

def extract_feature(feature_extractor, args):
    ''' read image data and labels from full_video_path'''
    transform = transforms.Compose([
                           transforms.Resize((224, 224)),
                           transforms.ToTensor(),   # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    '''read data'''
    category_list = sorted(os.listdir(os.path.join(args.full_video_path)))
    print("category_list: ",category_list)
    #all_imgs_list, all_labels_list = [], []
    all_imgs_list = []

    for category in category_list:
        img_dir = os.path.join(args.full_video_path, category)
        img_path = []
        for path in glob.glob(os.path.join(img_dir, '*.jpg')):
            img_path.append(path)
        img_path.sort()

        img_list = []
        for p in img_path:
            img = Image.open(p).convert('RGB')
            img = transform(img)
            img_list.append(img)
        all_imgs_list.append(torch.stack(img_list))

    """
    for category in category_list:
        label_list = []
        with open(os.path.join(args.full_video_label_path, category+'.txt')) as f:
            for line in f:
                label_list.append([int(line.strip('\n'))])
        label_list = np.array(label_list)
        all_labels_list.append(label_list)
    all_labels = np.array(all_labels_list)
    """

    '''extract features from image data'''
    feature_extractor.cuda()
    feature_extractor.eval()
    train_features = []
    with torch.no_grad():
        for idx in range(len(all_imgs_list)):
            ''' move data to gpu '''
            total_length = all_imgs_list[idx].shape[0]
            #print(total_length)
            features = []
            for i in range(0, total_length, args.extract_batch):
                if i+args.extract_batch > total_length:
                    imgs = all_imgs_list[idx][i:].cuda()
                else:
                    imgs = all_imgs_list[idx][i:i+args.extract_batch].cuda()
                #print("imgs.shape: ",imgs.shape)

                ''' forward path '''
                feature = feature_extractor(imgs)  #[sample #, 2048]
                features.append(feature.detach().cpu().numpy())
                torch.cuda.empty_cache()
            features = np.concatenate(features)
            train_features.append(features)

    train_features = np.array(train_features)

    #return train_features, all_labels, category_list
    return train_features, category_list


args = parser.arg_parse()
torch.cuda.set_device(args.gpu)

print('===> prepare model ...')
feature_extractor, seq2seq = models.P3(args)
''' load weight from P2 results '''
print('===> load model weight ...')
seq2seq.load_state_dict(torch.load('seq2seq_best.pth.tar', map_location="cuda:0"))

print('===> extract features ...')
#all_features, all_labels, category_list = extract_feature(feature_extractor, args)
all_features, category_list = extract_feature(feature_extractor, args)

seq2seq.cuda()
seq2seq.eval()


with torch.no_grad():
    video_num = all_features.shape[0]
    print(video_num)
    for vid_idx in range(video_num):
        total_length = all_features[vid_idx].shape[0]
        preds = []
        for idx in range(0, total_length, args.crop_batch):
            if idx+args.crop_batch > total_length:
                feature = all_features[vid_idx][idx:]
            else:
                feature = all_features[vid_idx][idx:idx+args.crop_batch]

            pred = seq2seq(torch.tensor(feature).unsqueeze(0).cuda())
            pred = torch.argmax(pred,1).detach().cpu().numpy()
            preds.append(pred)

        preds = np.concatenate(preds)

        with open(os.path.join(args.output_csv, '{}.txt'.format(category_list[vid_idx])), 'w') as f:
            for pred in preds:
                f.write('%d\n' % pred)


''' save ground truth as <category>_gt.txt
video_num = all_labels.shape[0]
for vid_idx in range(video_num):
    valid_labels = all_labels[vid_idx]
    print(valid_labels.shape)

    with open(os.path.join(args.output_csv, '{}_gt.txt'.format(category_list[vid_idx])), 'w') as f:
        for label in valid_labels:
            f.write('%d\n' % label)
'''

#python hw4_p3.py --full_video_path ../hw4_data/FullLengthVideos/videos/valid/ --output_csv p3_log/ --full_video_label_path ../hw4_data/FullLengthVideos/labels/valid
