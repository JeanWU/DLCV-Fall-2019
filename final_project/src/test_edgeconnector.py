import parser
import os
import torch
import data
import models
import utils
import loss
import math


if __name__ == '__main__':
    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_samples_dir = os.path.join(args.save_dir, 'samples')
    if not os.path.exists(save_samples_dir):
        os.makedirs(save_samples_dir)

    torch.cuda.set_device(args.gpu)

    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(data.DATA(args, mode='test'),
                                               batch_size=args.test_batch,
                                               num_workers=args.workers,
                                               shuffle=False)

    print('===> prepare model ...')
    unet = models.PConvUNet()
    unet = unet.cuda()
    discriminator = models.Discriminator(in_channels=3)
    discriminator = discriminator.cuda()

    if args.finetune:
        unet.freeze_enc_bn = True    # freeze bn layer for fine tuning
        optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr_finetune)
    else:
        optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    if args.resume:
        # start_iter = utils.load_ckpt(args.resume_folder, [('model', unet)])
        utils.load_ckpt(args.resume_folder, [('model', unet)])
        # unet.load_state_dict(torch.load(args.resume_folder))

    print('===> prepare loss function ...')
    criterion = loss.InpaintingLoss(models.VGG16FeatureExtractor()).cuda()

    print('===> prepare lambda ...')
    LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}

    print('===> start training ...')
    iters = 0
    best_score = 0

    for epoch in range(1, args.epoch+1):
        unet.train()

        for idx, (x, mask, gt, crop_box) in enumerate(train_loader):
            iters += 1
            x, gt, mask = x.cuda(), gt.cuda(), mask.cuda()
            a, b, h, w = crop_box
            b_size = x.shape[0]

            optimizer.zero_grad()
            I_out, output_mask = unet(x, mask)
            x = x[:, :, b:w, a:h]
            mask = mask[:, :, b:w, a:h]
            I_out = I_out[:, :, b:w, a:h]
            gt = gt[:, :, b:w, a:h]

            dis_real, dis_real_feat = discriminator(gt)        # in: (grayscale(1) + edge(1))
            dis_fake, dis_fake_feat = discriminator(I_out)

            loss_dict = criterion(x, mask, I_out, gt)
            loss = 0.0
            for key, coef in LAMBDA_DICT.items():
                value = coef * loss_dict[key]
                loss += value

            loss.backward()
            optimizer.step()

            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f] "
                  %(epoch, args.epoch, idx, len(train_loader), loss))

        if epoch % args.val_epoch == 0:
            mse, ssim = utils.evaluate(test_loader, unet, args)
            print("mse = {}, ssim = {}".format(mse, ssim))
            score = 1 - mse/100.0 + ssim
            if score > best_score:
            # if ssim > best_ssim:
                torch.save(unet.state_dict(), os.path.join(args.save_dir, 'unet_{}.pth'.format(epoch)))
                best_score = score
                # best_ssim = ssim
                utils.save_samples(test_loader, unet, save_samples_dir, epoch)

            print('ssim = {}, mse = {}, score = {}, best_score = {} '.format(ssim, mse, score, best_score))

    print('best_score = {} '.format(best_score))


'''
Reference:
https://github.com/ceshine/fast-neural-style/blob/master/notebooks/01-image-style-transfer.ipynb

'''
