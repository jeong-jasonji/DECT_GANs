#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from datasets import ImageDataset

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# line: python test.py --cuda cuda:1 --ckpt_path ./GAN_debias/train_A_1channel/

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataframeroot', type=str, default='merged_anno.csv', help='path to the csv dataframe')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', type=str, default='cuda:1', help='set cuda number e.g. cuda:1')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--ckpt_path', type=str, default='./output/', help='checkpoint path')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    device = torch.device(opt.cuda)

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.to(device)
    netG_B2A.to(device)

# Load state dicts
try:
    netG_A2B.load_state_dict(torch.load(opt.ckpt_path + 'output/netG_A2B_best.pth'))
    netG_B2A.load_state_dict(torch.load(opt.ckpt_path + 'output/netG_B2A_best.pth'))
except:
    netG_A2B.load_state_dict(torch.load(opt.ckpt_path + 'output/netG_A2B.pth'))
    netG_B2A.load_state_dict(torch.load(opt.ckpt_path + 'output/netG_B2A.pth'))
    
# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if 'cuda' in opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size).to(device)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size).to(device)

# Dataset loader
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

dataloader = DataLoader(ImageDataset(opt.dataframeroot, transforms_=transforms_, n_c=opt.input_nc, mode='test', unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists(opt.ckpt_path + 'output/A'):
    os.makedirs(opt.ckpt_path + 'output/A')
if not os.path.exists(opt.ckpt_path + 'output/B'):
    os.makedirs(opt.ckpt_path + 'output/B')
if not os.path.exists(opt.ckpt_path + 'output/mix'):
    os.makedirs(opt.ckpt_path + 'output/mix')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Save image files
    #save_image(fake_A, './output/A/%04d.png' % (i+1))
    #save_image(fake_B, './output/B/%04d.png' % (i+1))
    plt.imshow(fake_A.squeeze().cpu(), cmap='gray')
    plt.savefig(opt.ckpt_path + 'output/A/%04d.png' % (i+1))
    plt.close()
    plt.imshow(fake_B.squeeze().cpu(), cmap='gray')
    plt.savefig(opt.ckpt_path + 'output/B/%04d.png' % (i+1))
    plt.close()
    
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
    plt.savefig(os.path.join(opt.ckpt_path + 'output/mix/checkpoint_{}.png'.format(i+1)))
    plt.close()

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
