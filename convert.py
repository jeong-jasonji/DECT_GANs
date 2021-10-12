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

# line: python convert.py --cuda cuda:1 --verbose True

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataframeroot', type=str, default='cohort_10_anno.csv', help='path to the csv dataframe')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', type=str, default='cuda:1', help='set cuda number e.g. cuda:1')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--load_path', type=str, default='./train_C_RAdam/', help='path to the model checkpoints')
parser.add_argument('--save_path', type=str, default='/home/jupyter-jjjeon3/Collaboratory/GAN_debias/', help='path to save the converted images')
parser.add_argument('--verbose', type=bool, default=False, help='Add verbosity?')
opt = parser.parse_args()
print(opt)

### Define a image save function ###
def save_img(img, img_name, save_path):
    """input image is a tensor"""
    plt.axis('off')
    plt.imshow(img.squeeze().cpu(), cmap='gray')
    plt.savefig(save_path + '{}'.format(img_name), bbox_inches='tight', pad_inches=0)
    plt.close()

if opt.verbose:
    print('adding device')
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    device = torch.device(opt.cuda)

###### Definition of variables ######
if opt.verbose:
    print('defining networks')
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.to(device)
    netG_B2A.to(device)

# Load state dicts - need to set to load path
if opt.verbose:
    print('loading weights')
try:
    netG_A2B.load_state_dict(torch.load(opt.load_path + 'output/netG_A2B_best_e1_13.pth'))  # temporary specific model name
    netG_B2A.load_state_dict(torch.load(opt.load_path + 'output/netG_B2A_best_e1_13.pth'))
except:
    netG_A2B.load_state_dict(torch.load(opt.load_path + 'output/netG_A2B.pth'))
    netG_B2A.load_state_dict(torch.load(opt.load_path + 'output/netG_B2A.pth'))
    
# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
if opt.verbose:
    print('memory allocation')
Tensor = torch.cuda.FloatTensor if 'cuda' in opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size).to(device)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size).to(device)

# Dataset loader
if opt.verbose:
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

dataloader = DataLoader(ImageDataset(opt.dataframeroot, transforms_=transforms_, n_c=opt.input_nc, mode='full', unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
###################################

###### Converting ######
if opt.verbose:
    print('adding folder paths, if they do not exist')
# Create output dirs if they don't exist - need to set to save path
if not os.path.exists(os.path.join(opt.save_path, 'Original_256')):
    os.makedirs(os.path.join(opt.save_path, 'Original_256'))
if not os.path.exists(os.path.join(opt.save_path, 'Swap_256')):
    os.makedirs(os.path.join(opt.save_path, 'Swap_256'))

# set save paths
if opt.verbose:
    print('setting save paths')
orig_save_path = os.path.join(opt.save_path, 'Original_256/')
conv_save_path = os.path.join(opt.save_path, 'Swap_256/')

if opt.verbose:
    print('converting images')
for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))
    
    # get original image names
    real_A_fn = batch['A_fn'][0]
    real_B_fn = batch['B_fn'][0]

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # get converted image names
    fake_A_fn = batch['Ac_fn'][0]
    fake_B_fn = batch['Bc_fn'][0]
    
    # save original images to Original_256
    save_img(real_A, real_A_fn, orig_save_path)
    save_img(real_B, real_B_fn, orig_save_path)
    
    # save converted images to Swap_256
    save_img(fake_A, fake_A_fn, conv_save_path)
    save_img(fake_B, fake_B_fn, conv_save_path)
    
    print(real_A_fn, fake_A_fn, real_B_fn, fake_B_fn)
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################