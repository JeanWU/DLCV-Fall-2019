import os
import torch
import torch.nn as nn
import parser
import models
import data

from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

import numpy as np
import utils
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
    feature_extractor, BiRNN = models.P2(args)

    ''' extract feature '''
    #print('===> extract features for every videos ...')
    #utils.extract_feature_p2(feature_extractor, train_loader, val_loader, args)

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    BiRNN.cuda()
    optimizer = torch.optim.Adam(BiRNN.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = lr_scheduler.StepLR(optimizer, step_size=50)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' load train and val features '''
    print('===> load train and val features and labels ...')
    train_features, train_labels, valid_features, valid_labels = utils.load_features(args)

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):
        BiRNN.train()
        utils.set_requires_grad(BiRNN, True)
        total_length = train_features.shape[0]
        # shuffle
        perm_index = np.random.permutation(len(train_features))
        train_X_sfl = [train_features[i] for i in perm_index]
        train_y_sfl = np.array(train_labels)[perm_index]
        # construct training batch
        for index in range(0,total_length,args.train_batch):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, index+1, len(train_loader))
            iters += 1
            optimizer.zero_grad()
            if index+args.train_batch > total_length:
                input_X = train_X_sfl[index:]
                input_y = train_y_sfl[index:]
            else:
                input_X = train_X_sfl[index:index+args.train_batch]
                input_y = train_y_sfl[index:index+args.train_batch]

            # pad the sequence
            input_X, input_y, length = utils.single_batch_padding(input_X, input_y)
            input_X = input_X.cuda()
            output, _ = BiRNN(input_X, length)
            loss = criterion(output, input_y.squeeze(1).cuda())
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), iters)
            train_info += ' loss: {:.4f}'.format(loss.item())

            print(train_info)


        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = utils.evaluate_p2(BiRNN, valid_features, valid_labels, args)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))
            print('best ACC:{}'.format(best_acc))

            ''' save best model '''
            if acc > best_acc:
                utils.save_model(BiRNN, os.path.join(args.save_dir, 'BiRNN_best.pth.tar'))
                best_acc = acc

        sched.step()


''' record for best acc
#1 image size = (224, 224); mean, std = 0.5, 0.5; resnet 50, feature size = 2048 => best acc: 0.131
#2 image size = (224, 224); mean, std = 0.5, 0.5; resnet 50, feature size = 2048 => best acc: 0.252
#3 using hidden layer instead of output layer, hidden_size = 128 => best acc: 0.469
#4 hidden_size = 512 => best acc: 0.472
#5 hidden_size = 512, add more FC layers => best acc: 0.461
#6 hidden_size = 512, use FC from p1 => best acc: 0.457
#7 同6，add batch norm => best acc: 0.473
#8 同7, lr = 0.0001 => best acc: 0.477
#9 hidden_size = 1024 => best acc: 0.466
#10 hidden_size = 1024, lr=0.00001 => best acc: 0.499
#11 hidden_size = 512, lr = 0.00002 => best acc: 0.494
'''

# hidden_size = 512, lr = 0.00001 => best acc: 0.494
# hidden_size = 512, lr = 0.00001, w/o batch norm => best acc: 0.479
# hidden_size = 512, lr = 0.000001, w/o batch norm => best acc: 0.368

'''
reference:
https://gist.github.com/tokestermw/912042a85a1d53169c2dc7253dca55f6
'''
