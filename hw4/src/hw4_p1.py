#python hw4_p1.py TrimmedVideos/video/valid TrimmedVideos/label/gt_valid.csv ./output/
import data
import torch
import parser
import models
import os
import numpy as np


def extract_feature(feature_extractor, val_loader):
    feature_extractor.cuda()
    feature_extractor.eval()
    valid_features, valid_labels = [], []
    with torch.no_grad():
        for idx, (imgs, label) in enumerate(val_loader):
            ''' move data to gpu '''
            feature_extractor.cuda()
            imgs = imgs.squeeze().cuda()

            ''' forward path '''
            features = feature_extractor(imgs)
            features_mean = torch.mean(features, dim=0).detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            valid_features.append(features_mean)
            valid_labels.append(label)
            torch.cuda.empty_cache()

    valid_features, valid_labels = np.array(valid_features), np.array(valid_labels)
    return valid_features, valid_labels



args = parser.arg_parse()
print('===> prepare data loader ...')
val_loader = torch.utils.data.DataLoader(data.DATA(args),
                                       batch_size=args.data_batch,
                                       num_workers=args.workers,
                                       shuffle=False)
torch.cuda.set_device(args.gpu)

''' load model '''
print('===> prepare model ...')
feature_extractor, FC = models.P1()
FC.load_state_dict(torch.load('FC_best.pth.tar', map_location="cuda:0"))

print('===> extract features ...')
valid_features, valid_labels = extract_feature(feature_extractor, val_loader)

FC.cuda()
FC.eval()

preds = []
with torch.no_grad():
    total_length = valid_features.shape[0]
    for index in range(0,total_length,args.test_batch):
        if index+args.test_batch > total_length:
            input_X = valid_features[index:]
            input_y = valid_labels[index:]
        else:
            input_X = valid_features[index:index+args.test_batch]
            input_y = valid_labels[index:index+args.test_batch]
        input_X = torch.tensor(input_X).cuda()
        pred = FC(input_X)
        pred = torch.argmax(pred,1).detach().cpu().numpy()
        preds.append(pred)

preds = np.concatenate(preds)
#print(preds)

with open(os.path.join(args.output_csv, 'p1_valid.txt'), 'w') as f:
    for pred in preds:
        f.write('%d\n' % pred)

''' save ground truth as p1_gt.txt
print(valid_labels.shape)
print(valid_labels)
valid_labels = valid_labels.squeeze(1)
print(valid_labels)

with open(os.path.join(args.output_csv, 'p1_gt.txt'), 'w') as f:
    for label in valid_labels:
        f.write('%d\n' % label)
'''

#python hw4_p1.py --video_path ../hw4_data/TrimmedVideos/video/valid/ --csv_path ../hw4_data/TrimmedVideos/label/gt_valid.csv --output_csv p1_log/
