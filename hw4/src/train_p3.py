import utils
import parser
import os
import torch
import models
import torch.nn as nn

from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter


if __name__=='__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' load model '''
    print('===> prepare model ...')
    feature_extractor, seq2seq = models.P3(args)
    ''' load weight from P2 results '''
    print('===> load model weight ...')
    seq2seq.load_state_dict(torch.load(args.load_model_path))

    ''' load dataset and extract features '''
    #print('===> extract features and labels ...')
    #utils.extract_feature_p3(feature_extractor, args)

    ''' load train and val features '''
    print('===> load train and val features and labels ...')
    train_features, train_labels, valid_features, valid_labels = utils.load_features(args)

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    seq2seq.cuda()
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = lr_scheduler.StepLR(optimizer, step_size=50)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):
        seq2seq.train()
        utils.set_requires_grad(seq2seq, True)

        ''' shuffle -> not yet '''
        video_num = train_features.shape[0]
        for vid_idx in range(video_num):
            print("video index: ", vid_idx)
            features = train_features[vid_idx]
            labels = train_labels[vid_idx]
            total_length = features.shape[0]
            for idx in range(0, total_length, args.crop_batch):
                train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, total_length)
                iters += 1
                if idx+args.crop_batch > total_length:
                    feature = features[idx:]
                    label = labels[idx:]
                else:
                    feature = features[idx:idx+args.crop_batch]
                    label = labels[idx:idx+args.crop_batch]

                pred = seq2seq(torch.tensor(feature).unsqueeze(0).cuda())
                loss = criterion(pred, torch.tensor(label.squeeze(1)).cuda())
                '''
                pred shape : [batch #, class num]
                label shape: [batch #]
                '''
                loss.backward()
                optimizer.step()

                writer.add_scalar('loss', loss.item(), iters)
                train_info += ' loss: {:.4f}'.format(loss.item())

                print(train_info)

        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = utils.evaluate_p3(seq2seq, valid_features, valid_labels, args)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))
            print('best ACC:{}'.format(best_acc))

            ''' save best model '''
            if acc > best_acc:
                utils.save_model(seq2seq, os.path.join(args.save_dir, 'seq2seq_best.pth.tar'))
                best_acc = acc

            sched.step()

# 100 epoch, best acc: 0.588
