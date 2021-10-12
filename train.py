#!/usr/bin/python3
import os
import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# trying out this new RAdam optimizer
import torch_optimizer as t_optim

# sample code: python train.py --cuda cuda:3 --optim radam --batchSize 1 --input_nc 1 --output_nc 1 --savePath ./train_C_RAdam --decay_epoch 1 --verbose True

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--savePath', type=str, default='./test', help='directory to save to')
parser.add_argument('--dataframeroot', type=str, default='merged_anno.csv', help='path to the csv dataframe')
parser.add_argument('--d_lr', type=float, default=0.0002, help='initial learning rate for discriminator')
parser.add_argument('--g_lr', type=float, default=0.0002, help='initial learning rate for generator')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--optim', type=str, default='adam', help='optimizer to use')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', type=str, default='cuda:1', help='set cuda number e.g. cuda:1')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--verbose', type=str, default='False', help='be verbose or not')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    device = torch.device(opt.cuda)
    
if opt.verbose == 'True':
    verb = True
else:
    verb = False

###### Definition of variables ######
# Networks
if verb:
    print('Generating Networks')
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if 'cuda' in opt.cuda:
    if verb:
        print('setting networks to GPU')
    netG_A2B.to(device)  # changed from .cuda() to .to(device) - double check
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

if verb:
    print('initializing weights')
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
if verb:
    print('setting up losses')
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
if verb:
    print('setting up optimizers and LR schedulers')
    
if opt.optim == 'radam':
    if verb:
        print('setting RAdam')
    optimizer_G = t_optim.RAdam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.g_lr, betas=(0.5, 0.999))
    optimizer_D_A = t_optim.RAdam(netD_A.parameters(), lr=opt.d_lr, betas=(0.5, 0.999))
    optimizer_D_B = t_optim.RAdam(netD_B.parameters(), lr=opt.d_lr, betas=(0.5, 0.999))
else:
    if verb:
        print('setting Adam')
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.g_lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.d_lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.d_lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
if verb:
    print('Target memory allocations')
Tensor = torch.cuda.FloatTensor if 'cuda' in opt.cuda else torch.Tensor
if verb:
    print(Tensor)
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size).to(device)  # changed from nothing to .to(device) - double check
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size).to(device)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False).to(device)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
if verb:
    print('setting up dataloader')
if opt.input_nc != 1:
    transforms_ = [ transforms.Resize((opt.size, opt.size), Image.BICUBIC), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
else:
    transforms_ = [ transforms.Resize((opt.size, opt.size), Image.BICUBIC), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataframeroot, transforms_=transforms_, n_c=opt.input_nc, mode='train', unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
#logger = Logger(opt.n_epochs, len(dataloader))  # remove logger for now/testing purposes
if verb:
    print('setting up logs')
logger = open(os.path.join(opt.savePath, 'training_logs.txt'), 'a')
print(opt, file=logger)
###################################

# Create output dirs if they don't exist
if not os.path.exists(os.path.join(opt.savePath, 'output/A')):
    os.makedirs(os.path.join(opt.savePath, 'output/A'))
if not os.path.exists(os.path.join(opt.savePath, 'output/B')):
    os.makedirs(os.path.join(opt.savePath, 'output/B'))
    
###### Training ######
if verb:
    print('starting training')
best_total_loss = 99999.0
best_optimal_loss = 99999.0
for epoch in range(opt.epoch, opt.n_epochs):
    # keeping track of % done
    p = 0
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        
        ### Calculate Total Loss for updating ### - need to update 
        optimal_loss = np.abs(loss_G.cpu().detach().numpy() - (loss_D_A.cpu().detach().numpy() + loss_D_B.cpu().detach().numpy()))
        total_loss = loss_G.cpu().detach().numpy() + (loss_D_A.cpu().detach().numpy() + loss_D_B.cpu().detach().numpy())
        
        if optimal_loss <= best_optimal_loss and total_loss <= best_total_loss:
            print('Epoch/Batch: {}/{}%\nloss_G: {}, loss_G_identity: {}\nloss_G_GAN: {}, loss_G_cycle: {}, loss_D: {}\n'.format(epoch, p, loss_G, (loss_identity_A + loss_identity_B), (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)), file=logger)
            print('Improved optimal loss (np.abs(loss_G - (loss_D_A + loss_D_B))) ({} -> {})\nImproved total loss (np.abs(loss_G + (loss_D_A + loss_D_B))) ({} -> {})\nUpdated weights\n'.format(best_optimal_loss, optimal_loss, best_total_loss, total_loss), file=logger)
            logger.flush()
            # Save figure
            if i > int((len(dataloader)/10)):
                f = plt.figure(figsize=(20,5))
                ax1 = f.add_subplot(1, 4, 1)
                plt.imshow(real_A.cpu().squeeze(), cmap='gray')
                plt.title('real_A')
                ax2 = f.add_subplot(1, 4, 2)
                plt.imshow(fake_B.cpu().squeeze(), cmap='gray')
                plt.title('fake_B')
                mse_AB = mean_squared_error(real_A.cpu().squeeze().numpy(), fake_B.cpu().squeeze().numpy())
                ssim_AB = ssim(real_A.cpu().squeeze().numpy(), fake_B.cpu().squeeze().numpy(), data_range=fake_B.cpu().squeeze().numpy().max() - fake_B.cpu().squeeze().numpy().min())
                ax2.set_xlabel(f'MSE: {mse_AB:.2f}, SSIM: {ssim_AB:.2f}')
                ax3 = f.add_subplot(1, 4, 3)
                plt.imshow(fake_A.cpu().squeeze(), cmap='gray')
                plt.title('fake_A')
                mse_BA = mean_squared_error(real_B.cpu().squeeze().numpy(), fake_A.cpu().squeeze().numpy())
                ssim_BA = ssim(real_B.cpu().squeeze().numpy(), fake_A.cpu().squeeze().numpy(), data_range=fake_A.cpu().squeeze().numpy().max() - fake_A.cpu().squeeze().numpy().min())
                ax3.set_xlabel(f'MSE: {mse_BA:.2f}, SSIM: {ssim_BA:.2f}')
                ax4 = f.add_subplot(1, 4, 4)
                plt.imshow(real_B.cpu().squeeze(), cmap='gray')
                plt.title('real_B')
                plt.savefig(os.path.join(opt.savePath, 'epoch_{}_checkpoint_{}.png'.format(epoch, i)))
                plt.clf()
            # Save model weights
            torch.save(netG_A2B.state_dict(), os.path.join(opt.savePath, 'output/netG_A2B_best.pth'))
            torch.save(netG_B2A.state_dict(), os.path.join(opt.savePath, 'output/netG_B2A_best.pth'))
            torch.save(netD_A.state_dict(), os.path.join(opt.savePath, 'output/netD_A_best.pth'))
            torch.save(netD_B.state_dict(), os.path.join(opt.savePath, 'output/netD_B_best.pth'))
            best_optimal_loss = optimal_loss
            best_total_loss = total_loss

        # Progress report (http://localhost:8097)
        #logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
         #           'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
         #           images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
        if i % int((len(dataloader)/100)) == 0:
            print('Epoch/Batch: {}/{}%\nloss_G: {}, loss_G_identity: {}\nloss_G_GAN: {}, loss_G_cycle: {}, loss_D: {}\n'.format(epoch, p, loss_G, (loss_identity_A + loss_identity_B), (loss_GAN_A2B + loss_GAN_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)), file=logger)
            logger.flush()
            p += 1
        if verb:
            print(i)
        # print every 10% the images
        if i % int((len(dataloader)/10)) == 0:
            # add structural similarity index and other metrics at the bottom
            f = plt.figure(figsize=(20,5))
            ax1 = f.add_subplot(1, 4, 1)
            plt.imshow(real_A.cpu().squeeze(), cmap='gray')
            plt.title('real_A')
            ax2 = f.add_subplot(1, 4, 2)
            plt.imshow(fake_B.cpu().squeeze(), cmap='gray')
            plt.title('fake_B')
            mse_AB = mean_squared_error(real_A.cpu().squeeze().numpy(), fake_B.cpu().squeeze().numpy())
            ssim_AB = ssim(real_A.cpu().squeeze().numpy(), fake_B.cpu().squeeze().numpy(), data_range=fake_B.cpu().squeeze().numpy().max() - fake_B.cpu().squeeze().numpy().min())
            ax2.set_xlabel(f'MSE: {mse_AB:.2f}, SSIM: {ssim_AB:.2f}')
            ax3 = f.add_subplot(1, 4, 3)
            plt.imshow(fake_A.cpu().squeeze(), cmap='gray')
            plt.title('fake_A')
            mse_BA = mean_squared_error(real_B.cpu().squeeze().numpy(), fake_A.cpu().squeeze().numpy())
            ssim_BA = ssim(real_B.cpu().squeeze().numpy(), fake_A.cpu().squeeze().numpy(), data_range=fake_A.cpu().squeeze().numpy().max() - fake_A.cpu().squeeze().numpy().min())
            ax3.set_xlabel(f'MSE: {mse_BA:.2f}, SSIM: {ssim_BA:.2f}')
            ax4 = f.add_subplot(1, 4, 4)
            plt.imshow(real_B.cpu().squeeze(), cmap='gray')
            plt.title('real_B')
            plt.savefig(os.path.join(opt.savePath, 'epoch_{}_checkpoint_{}.png'.format(epoch, i)))
            plt.clf()

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints - Added conditional checkpoint saving, if the losses are lower, update "best model"
    torch.save(netG_A2B.state_dict(), os.path.join(opt.savePath, 'output/netG_A2B.pth'))
    torch.save(netG_B2A.state_dict(), os.path.join(opt.savePath, 'output/netG_B2A.pth'))
    torch.save(netD_A.state_dict(), os.path.join(opt.savePath, 'output/netD_A.pth'))
    torch.save(netD_B.state_dict(), os.path.join(opt.savePath, 'output/netD_B.pth'))
    
logger.close()
###################################
