import parser
import os
import torch
import data
import models
from torchvision.utils import save_image
from PIL import Image
import numpy as np


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x


if __name__=='__main__':
    args = parser.arg_parse()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    torch.cuda.set_device(args.gpu)

    print('===> prepare dataloader ...')
    test_loader = torch.utils.data.DataLoader(data.TEST_DATA(args),
                                               batch_size=args.test_batch,
                                               num_workers=args.workers,
                                               shuffle=False)
    print('===> prepare model ...')
    unet = models.PConvUNet()
    unet = unet.cuda()
    unet.freeze_enc_bn = True
    # utils.load_ckpt(args.resume_folder, [('model', unet)])
    unet.load_state_dict(torch.load(args.resume_folder, map_location='cuda:0'))
    unet.eval()
    for idx, (x, mask, crop_box) in enumerate(test_loader):
        x, mask = x.cuda(), mask.cuda()
        a, b, h, w = crop_box

        I_out, output_mask = unet(x, mask)
        x = x[:, :, b:w, a:h]
        I_out = I_out[:, :, b:w, a:h]
        mask = mask[:, :, b:w, a:h]
        output = mask * x + (1 - mask) * I_out
        output = output.detach().cpu()
        output = unnormalize(output)
        save_image(output, os.path.join(args.save_dir, '{}.jpg'.format(401+idx)))

        '''
        info = np.info(data.dtype) # Get the information of the incoming image type
        data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
        data = 255 * data # Now scale by 255
        img = data.astype(np.uint8)
        '''


        '''
        print("output without multiple: ", output)
        output = Image.fromarray((output * 255).astype(np.uint8))
        print("output with multiple: ", output)
        # img = Image.fromarray(img.astype('uint8'))
        gt = gt.squeeze().permute(1,2,0).cpu().numpy()
        gt = Image.fromarray((gt * 255).astype(np.uint8))
        #output = unnormalize(output.detach().cpu())
        #gt = unnormalize(gt.cpu())
        '''
        #save_image(output, os.path.join(save_mask_img, '{}.jpg'.format(401+idx)))
        #save_image(gt, os.path.join(save_gt, '{}.jpg'.format(401+idx)))
