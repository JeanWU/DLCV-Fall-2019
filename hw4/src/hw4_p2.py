#python hw4_p2.py TrimmedVideos/video/valid TrimmedVideos/label/gt_valid.csv ./output/
import data
import torch
import parser
import models
import os
import numpy as np
import utils

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
            valid_features.append(features.detach().cpu().numpy())
            valid_labels.append(label.detach().cpu().numpy())
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
feature_extractor, BiRNN = models.P2(args)
BiRNN.load_state_dict(torch.load('BiRNN_best.pth.tar', map_location="cuda:0"))

print('===> extract features ...')
valid_features, valid_labels = extract_feature(feature_extractor, val_loader)

BiRNN.cuda()
BiRNN.eval()

preds, gt = [], []
with torch.no_grad():
    total_length = valid_features.shape[0]
    print("total_length: ",total_length)
    """ using batch size = 1, so no need to worry about padding and sequence length """
    for index in range(0,total_length):
        input_X = torch.tensor(valid_features[index])
        input_X, input_y, lengths = utils.single_batch_padding([input_X],[valid_labels[index]],test=True)

        input_X = input_X.cuda()
        pred, _ = BiRNN(input_X, lengths)
        pred = torch.argmax(pred,1).detach().cpu().numpy()
        preds.append(pred)

preds = np.concatenate(preds)


with open(os.path.join(args.output_csv, 'p2_result.txt'), 'w') as f:
    for pred in preds:
        f.write('%d\n' % pred)

''' save ground truth as p1_gt.txt
valid_labels = valid_labels.squeeze(1)

with open(os.path.join(args.output_csv, 'p2_gt.txt'), 'w') as f:
    for label in valid_labels:
        f.write('%d\n' % label)
'''
#python hw4_p2.py --video_path ../hw4_data/TrimmedVideos/video/valid/ --csv_path ../hw4_data/TrimmedVideos/label/gt_valid.csv --output_csv p2_log/
