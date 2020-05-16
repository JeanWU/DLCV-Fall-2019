import os
import torch
import parser
import data
from torch.optim import lr_scheduler
import torch.nn as nn
from tensorboardX import SummaryWriter
import models
import numpy as np


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

def evaluate(model, data_loader, args):
    ''' set model to evaluate mode '''
    torch.cuda.set_device(args.gpu)
    feature_extractor, label_predictor = model
    feature_extractor.eval(), label_predictor.eval()

    preds, gt = [], []
    with torch.no_grad(): # do not need to calculate information for gradient during eval
        for idx, (imgs, classes) in enumerate(data_loader):
            imgs = imgs.cuda()
            features = feature_extractor(imgs)
            class_preds = label_predictor(features.view(-1, 512))
            class_preds = torch.argmax(class_preds, dim=1)
            class_preds = class_preds.detach().cpu().numpy()
            classes = classes.detach().cpu().numpy()
            preds.append(class_preds)
            gt.append(classes)

    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    return sum(1 for x,y in zip(preds, gt) if x == y) / float(len(preds))

if __name__=='__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    src_train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset=args.source_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    tar_train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset=args.target_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    tar_test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.target_dataset),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

    ''' load model '''
    print('===> prepare model ...')
    feature_extractor, label_predictor, domain_classifier = models.DANN(args)
    feature_extractor.cuda(), label_predictor.cuda(), domain_classifier.cuda()

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(label_predictor.parameters()) \
                                 + list(domain_classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    sched = lr_scheduler.StepLR(optimizer, step_size=50)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):
        len_dataloader = min(len(src_train_loader), len(tar_train_loader))

        feature_extractor.train(), label_predictor.train(), domain_classifier.train()
        for idx, (src_data, tar_data) in enumerate(zip(src_train_loader, tar_train_loader)):
            iters += 1
            if args.dann_type == 'both':
                p = float(idx + epoch * len_dataloader) / args.epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                img1, label1 = src_data
                img2, label2 = tar_data
                size = min((img1.shape[0], img2.shape[0]))
                img1, label1 = img1[0:size, :, :, :], label1[0:size]
                img2, label2 = img2[0:size, :, :, :], label2[0:size]

                img1, label1, img2, label2 = img1.cuda(), label1.long().cuda(), img2.cuda(), label2.long().cuda()
                optimizer.zero_grad()

                src_domain_label = torch.zeros((img1.size()[0])).type(torch.LongTensor).cuda()
                tar_domain_label = torch.ones((img2.size()[0])).type(torch.LongTensor).cuda()

                src_feature = feature_extractor(img1)
                tar_feature = feature_extractor(img2)
                # compute class label loss
                class_preds = label_predictor(src_feature.view(-1, 512))
                class_label_loss = criterion(class_preds, label1)
                writer.add_scalar('class label loss', class_label_loss.data.cpu().numpy(), iters)
                # compute domain label loss
                tar_preds = domain_classifier(tar_feature.view(-1, 512), alpha)
                src_preds = domain_classifier(src_feature.view(-1, 512), alpha)
                tar_domain_loss = criterion(tar_preds, tar_domain_label)
                src_domain_loss = criterion(src_preds, src_domain_label)
                domain_loss = tar_domain_loss + src_domain_loss
                writer.add_scalar('domain loss', domain_loss.data.cpu().numpy(), iters)


                loss = class_label_loss + domain_loss
                loss.backward()
                optimizer.step()

                # print loss
                print('Epoch: [{}][{}/{}]\tLoss: {:.4f}\tClass Loss: {:.4f}\tDomain Loss:{:.4f}'.format(\
                    epoch, idx+1, len(src_train_loader), loss.item(), class_label_loss.item(),\
                              domain_loss.item()))

            # only train on single domain data
            if args.dann_type == 'single':
                img1, label1 = src_data
                img1, label1 = img1.cuda(), label1.long().cuda()
                optimizer.zero_grad()
                src_feature = feature_extractor(img1)
                class_preds = label_predictor(src_feature.view(-1, 512))
                class_label_loss = criterion(class_preds, label1)
                class_label_loss.backward()
                optimizer.step()

                print('Epoch: [{}][{}/{}]\tClass Loss: {:.4f}'.format(\
                    epoch, idx+1, len(src_train_loader), class_label_loss.item()))



        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate((feature_extractor, label_predictor), tar_test_loader, args)

            ''' save best model '''
            if acc > best_acc:
                save_model(feature_extractor, os.path.join(args.save_dir, 'best_feature_extractor.pth.tar'))
                save_model(label_predictor, os.path.join(args.save_dir, 'best_label_predictor.pth.tar'))
                save_model(domain_classifier, os.path.join(args.save_dir, 'best_domain_classifier.pth.tar'))
                best_acc = acc
