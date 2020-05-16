import data
import torch
import parser
import models
import csv

args = parser.arg_parse()


def saveResult(model, data_loader, save_dir):

    ''' set model to evaluate mode '''
    torch.cuda.set_device(args.gpu)
    feature_extractor, label_predictor = model
    feature_extractor.eval(), label_predictor.eval()

    result_dict = {}
    with torch.no_grad(): # do not need to calculate information for gradient during eval
        for idx, (imgs, img_name) in enumerate(data_loader):
            imgs = imgs.cuda()
            features = feature_extractor(imgs)
            class_preds = label_predictor(features.view(-1, 512))
            pred = torch.argmax(class_preds, dim=1)
            pred_list = pred.tolist()
            img_name_list = list(img_name)
            for idx, item in enumerate(pred_list):
                result_dict[img_name_list[idx]]=item

    with open(save_dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'label'])
        for key in result_dict.keys():
            writer.writerow([key, result_dict[key]])


''' setup GPU '''
torch.cuda.set_device(args.gpu)

''' prepare data_loader '''
data_loader = torch.utils.data.DataLoader(data.TESTDATA(args),
                                              batch_size=args.train_batch,
                                              num_workers=args.workers,
                                              shuffle=False)
''' prepare model '''
tar_feature_extractor, label_predictor, _ = models.ADDA(args)
tar_feature_extractor.cuda(), label_predictor.cuda()

''' resume save model '''
if args.target_domain == 'mnistm':
    tar_feature_extractor.load_state_dict(torch.load('s2m_adda_feature_extractor.pth.tar',map_location='cuda:0'))
    label_predictor.load_state_dict(torch.load('s2m_adda_label_predictor.pth.tar',map_location='cuda:0'))
elif args.target_domain == 'svhn':
    tar_feature_extractor.load_state_dict(torch.load('m2s_adda_feature_extractor.pth.tar',map_location='cuda:0'))
    label_predictor.load_state_dict(torch.load('m2s_adda_label_predictor.pth.tar',map_location='cuda:0'))

saveResult((tar_feature_extractor, label_predictor), data_loader, args.save_folder)


