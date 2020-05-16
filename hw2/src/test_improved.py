import os
import parser
from modeling.deeplab import *
import data
import cv2
import numpy as np

def mean_iou_score(pred, labels, num_classes=9):
    '''
    Compute mean IoU score over 9 classes
    '''
    mean_iou = 0
    for i in range(num_classes):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / num_classes
    return mean_iou

def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    segs = []
    with torch.no_grad(): # do not need to calculate information for gradient during eval
        for idx, (imgs, seg, img_path) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
            pred = torch.argmax(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()
            seg = seg.numpy().squeeze()

            preds.append(pred)
            segs.append(seg)

    segs = np.concatenate(segs)
    preds = np.concatenate(preds)

    return mean_iou_score(pred=preds, labels=segs)

def saveResult(model, data_loader):
    model.eval()
    pred_folder = 'pred_result_improved'
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)
    with torch.no_grad(): # do not need to calculate information for gradient during eval
        for idx, (imgs, seg, img_path) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)
            pred = torch.argmax(pred, dim=1)    #batch size = 1
            pred = pred.detach().cpu().numpy()
            filename = ''.join(img_path)
            cv2.imwrite(os.path.join(pred_folder, filename[-8:]), pred[0])

if __name__ == '__main__':

    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              shuffle=True)
    ''' prepare model '''
    model = DeepLab(num_classes=9,
                    backbone='resnet',
                    output_stride=16,
                    freeze_bn=False)
    model.cuda() # load model to gpu

    ''' resume save model '''
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

    acc = evaluate(model, test_loader)
    print('Testing Accuracy: {}'.format(acc))
    saveResult(model, test_loader)
