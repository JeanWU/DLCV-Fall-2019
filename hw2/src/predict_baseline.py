import parser
import torch
import baseline_model
import data
import cv2
import os

def saveResult(model, data_loader, pred_folder):
    model.eval()
    with torch.no_grad(): # do not need to calculate information for gradient during eval
        for idx, (imgs, img_path) in enumerate(data_loader):
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
    test_loader = torch.utils.data.DataLoader(data.TESTDATA(args),
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              shuffle=True)
    ''' prepare model '''
    model = baseline_model.Net(args).cuda()

    ''' resume save model '''
    checkpoint = torch.load(args.resume, map_location='cuda:0')
    model.load_state_dict(checkpoint)
    saveResult(model, test_loader, args.save_predict)
