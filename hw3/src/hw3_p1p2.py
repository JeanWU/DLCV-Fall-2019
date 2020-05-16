import parser
import numpy as np
import torch
import models
from torchvision.utils import save_image
import os
import sys

args = parser.arg_parse()
device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available()) else "cpu")
torch.cuda.set_device(args.gpu)

''' setup random seed '''
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

''' create fixed noise for p1 and p2 '''
p1_noise = torch.randn(32, args.nz, 1, 1, device=device)

fixed_noise = torch.randn(10, args.nz, 1, 1, device=device)
fixed_noise_1 = torch.cat((fixed_noise, torch.zeros((10, 1, 1, 1), device=device)), dim=1)
fixed_noise_2 = torch.cat((fixed_noise, torch.ones((10, 1, 1, 1), device=device)), dim=1)
p2_noise = torch.cat((fixed_noise_1, fixed_noise_2), dim=0)

''' load model '''

dc_netG, dc_netD = models.GAN(args)
dc_netG.cuda().eval()
dc_netG.load_state_dict(torch.load('best_dc_negG.pth.tar',map_location='cuda:0'))

ac_netG, ac_netD = models.ACGAN(args)
ac_netG.cuda().eval()
ac_netG.load_state_dict(torch.load('best_ac_negG.pth.tar',map_location='cuda:0'))


''' generate images '''
with torch.no_grad():

    fake_p1 = dc_netG(p1_noise).detach().cpu()
    save_image(fake_p1.data, os.path.join(args.save_dir, "fig1_2.jpg"), nrow=5, normalize=True)


    fake_p2 = ac_netG(p2_noise).detach().cpu()
    save_image(fake_p2.data, os.path.join(args.save_dir, "fig2_2.jpg"), nrow=10, normalize=True)
