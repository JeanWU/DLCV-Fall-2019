import models
import torch
import numpy as np
from skimage.metrics import structural_similarity
import os
from torchvision.utils import save_image
import evaluate as eva


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(STD) + torch.Tensor(MEAN)
    x = x.transpose(1, 3)
    return x


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, torch.nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']


def evaluate(dataloader, unet, args):
    valid_dir = os.path.join(args.save_dir, 'valid_mask')
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    unet.cuda()
    unet.eval()

    for idx, (x, mask, gts, crop_box) in enumerate(dataloader):
        x, mask = x.cuda(), mask.cuda()
        a, b, h, w = crop_box
        I_out, output_mask = unet(x, mask)
        output = mask * x + (1 - mask) * I_out
        output = output[:, :, b:w, a:h]
        output = output.detach().cpu()
        output = unnormalize(output)
        save_image(output, os.path.join(valid_dir, '{}.jpg'.format(401+idx)))

    img_gt_paths = []
    img_pred_paths = []
    for i in range(100):
        img_name = "{}.jpg".format(401 + i)
        img_gt_paths.append(os.path.join(args.ground_truth, img_name))
        img_pred_paths.append(os.path.join(valid_dir, img_name))

    mse, ssim = eva.get_average_mse_ssim(img_gt_paths, img_pred_paths)

    return mse, ssim


def save_samples(dataloader, unet, save_samples_dir, epoch):
    unet.cuda()
    unet.eval()
    for idx, (x, mask, gts, crop_box) in enumerate(dataloader):
        x, mask = x.cuda(), mask.cuda()
        a, b, h, w = crop_box
        I_out, output_mask = unet(x, mask)
        x = x[:, :, b:w, a:h]
        I_out = I_out[:, :, b:w, a:h]
        mask = mask[:, :, b:w, a:h]
        output_comp = mask * x + (1 - mask) * I_out
        gts = gts[:, :, b:w, a:h]
        imgs = torch.cat((gts.cpu(), x.cpu(), I_out.cpu(), output_comp.cpu()), dim=0)
        save_image(imgs, os.path.join(save_samples_dir, 'epoch_{}_{}.jpg'.format(epoch, idx)), nrow=len(x))
        if idx > 0:
            break
