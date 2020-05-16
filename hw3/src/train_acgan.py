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
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    ''' load model '''
    print('===> prepare model ...')
    netG, netD = models.ACGAN(args)
    netG, netD = netG.cuda(), netD.cuda()

    ''' define loss '''
    adversarial_loss = nn.BCELoss()
    auxiliary_loss = nn.CrossEntropyLoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(args.samples_num, args.nz, 1, 1, device=device)
    fixed_noise_1 = torch.cat((fixed_noise, torch.zeros((args.samples_num, 1, 1, 1), device=device)), dim=1)
    fixed_noise_2 = torch.cat((fixed_noise, torch.ones((args.samples_num, 1, 1, 1), device=device)), dim=1)
    fixed_noise = torch.cat((fixed_noise_1, fixed_noise_2), dim=0)


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
            real_img = imgs.cuda()
            b_size = real_img.size(0)

            # -----------------
            #  Train Generator
            # -----------------
            optimizerG.zero_grad()
            label_real = torch.full((b_size,), real_label, device=device)
            # Sample noise and labels as generator input
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)  # +1 for class
            noise = torch.cat((noise, classes.view(b_size, 1, 1, 1).to(device)), dim=1)
            # Generate a batch of images
            gen_imgs = netG(noise)
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = netD(gen_imgs)
            print("classes shape: ", classes.shape) # should be [batch_size, 1]
            g_loss = 0.5 * (adversarial_loss(validity, label_real) + auxiliary_loss(pred_label, classes.long().cuda()))

            g_loss.backward()
            optimizerG.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizerD.zero_grad()
            # Loss for real images
            real_pred, real_aux = netD(real_img)
            d_real_loss = (adversarial_loss(real_pred, label_real) + auxiliary_loss(real_aux, classes.long().cuda())) / 2

            label_fake = torch.full((b_size,), fake_label, device=device)
            # Loss for fake images
            fake_pred, fake_aux = netD(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, label_fake) + auxiliary_loss(fake_aux, classes.long().cuda())) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([classes.view(-1).data.cpu().numpy(), classes.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            optimizerD.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                % (epoch, args.epoch, idx, len(train_loader), d_loss.item(), 100 * d_acc, g_loss.item())
            )

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.epoch-1) and (idx == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    save_image(fake.data, os.path.join(args.save_dir, args.save_samples_dir, "fake_%d.png" % iters), nrow=10, normalize=True)

                save_model(netG, os.path.join(args.save_dir, 'netG_{}.pth.tar'.format(iters)))
                save_model(netD, os.path.join(args.save_dir, 'netD_{}.pth.tar'.format(iters)))
