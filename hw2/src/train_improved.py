import os
import parser
from modeling.deeplab import *
import data
from utils.loss import SegmentationLosses
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from test import evaluate

def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

if __name__=='__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_improved_dir):
        os.makedirs(args.save_improved_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    model = DeepLab(num_classes=9,
                    backbone='resnet',
                    output_stride=16,
                    freeze_bn=False)
    model.cuda() # load model to gpu

    ''' define loss '''
    criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode=args.loss_type)

    ''' setup optimizer '''
    train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                    {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
    optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    sched = lr_scheduler.StepLR(optimizer, step_size=30)
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_improved_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):

        model.train()

        for idx, (imgs, seg) in enumerate(train_loader):

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            imgs, seg = imgs.cuda(), seg.cuda()

            ''' forward path '''
            output = model(imgs)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, seg) # compute loss

            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)

        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(model, val_loader)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(model, os.path.join(args.save_improved_dir, 'model_best.pth.tar'))
                best_acc = acc

        ''' save model '''
        #save_model(model, os.path.join(args.save_improved_dir, 'model_{}.pth.tar'.format(epoch)))
        sched.step()
