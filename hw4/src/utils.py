import torch
import os
import numpy as np
import torchvision.transforms as transforms
import glob
from PIL import Image

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def extract_feature_p1(feature_extractor, train_loader, val_loader, args):
    feature_extractor.cuda()
    feature_extractor.eval()
    train_features, train_labels, valid_features, valid_labels = [], [], [], []
    with torch.no_grad():
        for idx, (imgs, label) in enumerate(train_loader):
            ''' move data to gpu '''
            feature_extractor.cuda()
            imgs = imgs.squeeze().cuda()

            ''' forward path '''
            features = feature_extractor(imgs)
            features_mean = torch.mean(features, dim=0).detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            train_features.append(features_mean)
            train_labels.append(label)
            torch.cuda.empty_cache()

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

    train_features, train_labels = np.array(train_features), np.array(train_labels)
    valid_features, valid_labels = np.array(valid_features), np.array(valid_labels)
    print("train_features.shape: ", train_features.shape)
    print("train_labels.shape: ", train_labels.shape)
    print("valid_features.shape: ", valid_features.shape)
    print("valid_labels.shape: ", valid_labels.shape)

    np.save(os.path.join(args.save_dir, "train_features.npy"), train_features)
    np.save(os.path.join(args.save_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(args.save_dir, "valid_features.npy"), valid_features)
    np.save(os.path.join(args.save_dir, "valid_labels.npy"), valid_labels)


def load_features(args):
    train_features = np.load(os.path.join(args.save_dir, "train_features.npy"))
    train_labels = np.load(os.path.join(args.save_dir, "train_labels.npy"))
    valid_features = np.load(os.path.join(args.save_dir, "valid_features.npy"))
    valid_labels = np.load(os.path.join(args.save_dir, "valid_labels.npy"))

    return train_features, train_labels, valid_features, valid_labels

def evaluate_p1(FC, valid_features, valid_labels, args):
    FC.cuda()
    FC.eval()

    preds, gt = [], []
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
            gt.append(input_y)

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    return sum(1 for x,y in zip(preds, gt) if x == y) / float(len(preds))

def extract_feature_p2(feature_extractor, train_loader, val_loader, args):
    feature_extractor.cuda()
    feature_extractor.eval()
    train_features, valid_features, train_labels, valid_labels = [], [], [], []
    with torch.no_grad():
        for idx, (imgs, label) in enumerate(train_loader):
            ''' move data to gpu '''
            print("idx: {}".format(idx))
            feature_extractor.cuda()
            imgs = imgs.squeeze().cuda()

            ''' forward path '''
            features = feature_extractor(imgs)  #[sample #, 2048]
            train_features.append(features.detach().cpu().numpy())
            train_labels.append(label.detach().cpu().numpy())
            torch.cuda.empty_cache()

        for idx, (imgs, label) in enumerate(val_loader):
            ''' move data to gpu '''
            print("idx: {}".format(idx))
            feature_extractor.cuda()
            imgs = imgs.squeeze().cuda()

            ''' forward path '''
            features = feature_extractor(imgs)
            valid_features.append(features.detach().cpu().numpy())
            valid_labels.append(label.detach().cpu().numpy())
            torch.cuda.empty_cache()

    train_features, valid_features = np.array(train_features), np.array(valid_features)
    train_labels, valid_labels = np.array(train_labels), np.array(valid_labels)
    print("train_features.shape: ",train_features.shape) #[2653, sample#, 40960]
    print("valid_features.shape: ",valid_features.shape)
    print("train_labels.shape: ", train_labels.shape)
    print("valid_labels.shape: ", valid_labels.shape)

    np.save(os.path.join(args.save_dir, "train_features.npy"), train_features)
    np.save(os.path.join(args.save_dir, "valid_features.npy"), valid_features)
    np.save(os.path.join(args.save_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(args.save_dir, "valid_labels.npy"), valid_labels)

def single_batch_padding(train_X_batch, train_y_batch, test = False):
    if test==True:
        padded_sequence = torch.nn.utils.rnn.pad_sequence(train_X_batch, batch_first=True)
        label = torch.LongTensor(train_y_batch)
        length = [len(train_X_batch[0])]
    else:
        length = [len(x) for x in train_X_batch]
        perm_index = np.argsort(length)[::-1]

        # sort by sequence length
        train_X_batch = [torch.tensor(train_X_batch[i]) for i in perm_index]
        length = [len(x) for x in train_X_batch]
        padded_sequence = torch.nn.utils.rnn.pad_sequence(train_X_batch, batch_first=True)
        label = torch.LongTensor(np.array(train_y_batch)[perm_index])
    return padded_sequence, label, length

def evaluate_p2(BiRNN, valid_features, valid_labels, args):
    BiRNN.cuda()
    BiRNN.eval()

    preds, gt = [], []
    with torch.no_grad():
        total_length = valid_features.shape[0]
        for index in range(0,total_length,args.test_batch):
            if index+args.test_batch > total_length:
                input_X = valid_features[index:]
                input_y = valid_labels[index:]
            else:
                input_X = valid_features[index:index+args.test_batch]
                input_y = valid_labels[index:index+args.test_batch]

            input_X, input_y, length = single_batch_padding(input_X, input_y)
            input_X = input_X.cuda()
            pred, _ = BiRNN(input_X, length)
            pred = torch.argmax(pred,1).detach().cpu().numpy()
            preds.append(pred)
            gt.append(input_y)

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    return sum(1 for x,y in zip(preds, gt) if x == y) / float(len(preds))


def extract_feature_p3(feature_extractor, args):
    ''' read image data and labels from full_video_path'''
    transform = transforms.Compose([
                           transforms.Resize((224, 224)),
                           transforms.ToTensor(),   # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    '''read data'''
    for mode in ['train','valid']:
        category_list = sorted(os.listdir(os.path.join(args.full_video_path, mode)))
        print("category_list: ",category_list)
        all_imgs_list, all_labels_list = [], []

        for category in category_list:
            img_dir = os.path.join(args.full_video_path, mode, category)
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


        for category in category_list:
            label_list = []
            with open(os.path.join(args.full_video_label_path, mode, category+'.txt')) as f:
                for line in f:
                    label_list.append([int(line.strip('\n'))])
            label_list = np.array(label_list)
            all_labels_list.append(label_list)
        all_labels_list = np.array(all_labels_list)


        '''extract features from image data'''
        feature_extractor.cuda()
        feature_extractor.eval()
        train_features = []
        with torch.no_grad():
            for idx in range(len(all_imgs_list)):
                ''' move data to gpu '''
                total_length = all_imgs_list[idx].shape[0]
                print(total_length)
                features = []
                for i in range(0, total_length, args.extract_batch):
                    if i+args.extract_batch > total_length:
                        imgs = all_imgs_list[idx][i:].cuda()
                    else:
                        imgs = all_imgs_list[idx][i:i+args.extract_batch].cuda()
                    print("imgs.shape: ",imgs.shape)

                    ''' forward path '''
                    feature = feature_extractor(imgs)  #[sample #, 2048]
                    features.append(feature.detach().cpu().numpy())
                    torch.cuda.empty_cache()
                features = np.concatenate(features)
                train_features.append(features)

        train_features = np.array(train_features)

        np.save(os.path.join(args.save_dir, "{}_features.npy".format(mode)), train_features)
        np.save(os.path.join(args.save_dir, "{}_labels.npy".format(mode)), all_labels_list)


def evaluate_p3(seq2seq, valid_features, valid_labels, args):
    seq2seq.cuda()
    seq2seq.eval()

    preds, gt = [], []
    with torch.no_grad():
        video_num = valid_features.shape[0]
        for vid_idx in range(video_num):
            total_length = valid_features[vid_idx].shape[0]
            for idx in range(0, total_length, args.crop_batch):
                if idx+args.crop_batch > total_length:
                    feature = valid_features[vid_idx][idx:]
                    label = valid_labels[vid_idx][idx:]
                else:
                    feature = valid_features[vid_idx][idx:idx+args.crop_batch]
                    label = valid_labels[vid_idx][idx:idx+args.crop_batch]

                pred = seq2seq(torch.tensor(feature).unsqueeze(0).cuda())
                pred = torch.argmax(pred,1).detach().cpu().numpy()
                preds.append(pred)
                gt.append(label)

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    return sum(1 for x,y in zip(preds, gt) if x == y) / float(len(preds))
