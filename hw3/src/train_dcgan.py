import os
import torch

import parser
import models
import data

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from tensorboardX import SummaryWriter


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)



if __name__=='__main__':

    args = parser.arg_parse()

    device = torch.device("cuda:" + str(args.gpu) if (torch.cuda.is_available()) else "cpu")

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    ''' load model '''
    print('===> prepare model ...')
    netG, netD = models.GAN(args)
    netG, netD = netG.cuda(), netD.cuda()

    ''' define loss '''
    adversarial_loss = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(args.samples_num, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    ''' setup optimizer '''
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_D, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_G, betas=(args.beta1, 0.999))

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    G_losses = []
    D_losses = []

    for epoch in range(1, args.epoch+1):

        netD.train()
        netG.train()

        for idx, (imgs, classes) in enumerate(train_loader):
            iters += 1

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_img = imgs.cuda()
            b_size = real_img.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            real_pred = netD(real_img).view(-1)
            errD_real = adversarial_loss(real_pred, label)
            errD_real.backward()
            D_x = real_pred.mean().item()
            writer.add_scalar('errD_real', errD_real.data.cpu().numpy(), iters)

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            fake_pred = netD(fake.detach()).view(-1)
            errD_fake = adversarial_loss(fake_pred, label)
            errD_fake.backward()
            D_G_z1 = fake_pred.mean().item()
            writer.add_scalar('errD_fake', errD_fake.data.cpu().numpy(), iters)

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            fake_pred = netD(fake).view(-1)
            errG = adversarial_loss(fake_pred, label)
            errG.backward()
            D_G_z2 = fake_pred.mean().item()
            optimizerG.step()
            writer.add_scalar('g loss', errG.data.cpu().numpy(), iters)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.epoch-1) and (idx == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    save_image(fake.data, os.path.join(args.save_dir, args.save_samples_dir, "fake_%d.png" % iters), nrow=5, normalize=True)

                save_model(netG, os.path.join(args.save_dir, 'netG_{}.pth.tar'.format(iters)))
                save_model(netD, os.path.join(args.save_dir, 'netD_{}.pth.tar'.format(iters)))



            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, args.epoch, idx, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
