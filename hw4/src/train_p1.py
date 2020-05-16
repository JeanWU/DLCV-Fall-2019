import os
import torch
import torch.nn as nn
import parser
import models
import data
import utils

from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import numpy as np
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

if __name__=='__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.data_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test'),
                                               batch_size=args.data_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

    ''' load model '''
    print('===> prepare model ...')
    feature_extractor, FC = models.P1()

    ''' extract feature '''
    #print('===> extract features for every videos ...')
    #utils.extract_feature_p1(feature_extractor, train_loader, val_loader, args)

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    FC.cuda()
    optimizer = torch.optim.Adam(FC.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = lr_scheduler.StepLR(optimizer, step_size=50)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' load train and val features '''
    print('===> load train and val features and labels ...')
    train_features, train_label, valid_features, valid_labels = utils.load_features(args)

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):
        FC.train()
        utils.set_requires_grad(FC, True)
        total_length = train_features.shape[0]
        perm_index = torch.randperm(total_length)
        train_X_sfl = train_features[perm_index]
        train_y_sfl = train_label[perm_index]
        # construct training batch
        for index in range(0,total_length,args.train_batch):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, index+1, len(train_loader))
            iters += 1
            optimizer.zero_grad()
            if index+args.train_batch > total_length:
                #break
                input_X = train_X_sfl[index:]
                input_y = train_y_sfl[index:]
            else:
                input_X = train_X_sfl[index:index+args.train_batch]
                input_y = train_y_sfl[index:index+args.train_batch]

            input_X = torch.tensor(input_X).cuda()
            output = FC(input_X)
            loss = criterion(output, torch.tensor(input_y.squeeze(1)).cuda())
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), iters)
            train_info += ' loss: {:.4f}'.format(loss.item())

            print(train_info)


        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = utils.evaluate_p1(FC, valid_features, valid_labels, args)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))
            print('best ACC:{}'.format(best_acc))

            ''' save best model '''
            if acc > best_acc:
                utils.save_model(FC, os.path.join(args.save_dir, 'FC_best.pth.tar'))
                best_acc = acc

        sched.step()

# run 100 epoch => best acc: 0.36
# run 100 epoch => best acc: 0.374 (dense net)
# run 100 epoch => best acc: 0.339
''' record for best acc
#1 image size = (120, 160); mean, std = 0.5, 0.5; resnet 50, feature size = 40960 => best acc: 0.371
#2 image size = (224, 224); mean, std = 0.5, 0.5; resnet 50, feature size = 2048 => best acc: 0.4369

'''
