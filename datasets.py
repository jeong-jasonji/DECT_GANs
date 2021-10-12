import glob
import random
import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, n_c=1, mode='train', unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.nc = n_c
        
        # make it load the data from the dataframe not a file list
        dataframe = pd.read_csv(root)
        
        #total length:
        full_len = len(dataframe['png_path'][dataframe['DEMO_NUM'] == 1].to_list())
        train_len = int(full_len*0.8)
        
        if mode == 'train':
            self.files_A = dataframe['png_path'][dataframe['DEMO_NUM'] == 1].to_list()[:train_len] # temp make only 100
            self.files_B = dataframe['png_path'][dataframe['DEMO_NUM'] == 0].to_list()[:train_len] # temp make only 100
        elif mode == 'full':
            self.files_A = dataframe['png_path'][dataframe['DEMO_NUM'] == 1].to_list() # temp make only 100
            self.files_B = dataframe['png_path'][dataframe['DEMO_NUM'] == 0].to_list() # temp make only 100
        else:
            self.files_A = dataframe['png_path'][dataframe['DEMO_NUM'] == 1].to_list()[train_len:]  # list of white patients' paths
            self.files_B = dataframe['png_path'][dataframe['DEMO_NUM'] == 0].to_list()[train_len:full_len]  # list of black patients' paths and make it the same length as white
            
        # get original filenames of each dataset
        self.files_A_orig_names = [i.split('/')[-1] for i in self.files_A]
        self.files_B_orig_names = [i.split('/')[-1] for i in self.files_B]
        # get converted filenames of each dataset
        self.files_A_conv_names = [i.split('/')[-1].replace('.png', '_c.png') for i in self.files_A]
        self.files_B_conv_names = [i.split('/')[-1].replace('.png', '_c.png') for i in self.files_B]
        
    def __getitem__(self, index):
        # change a grayscale image to stacked rgb for A
        image_a = Image.open(self.files_A[index % len(self.files_A)])  # selects a file from index and opens with PIL
        if self.nc != 1:
            img_a_array = np.array(image_a)  # convert to numpy array
            img_a_3ch = np.stack((img_a_array,) * 3, axis=-1)  # convert to a 3D stack
            item_A = self.transform(Image.fromarray(img_a_3ch, mode='RGB'))  # apply the transforms
        else:
            img_a_array = np.array(image_a).astype('float').copy()
            img_a_norm = img_a_array / img_a_array.max()
            item_A = self.transform(Image.fromarray(img_a_norm))
    
        # change a grayscale image to stacked rgb for B but with aligned/unaligned conditions
        if self.unaligned:
            image_b = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_b = Image.open(self.files_B[index % len(self.files_B)])
        if self.nc != 1:
            img_b_array = np.array(image_b)
            img_b_3ch = np.stack((img_b_array,) * 3, axis=-1)
            item_B = self.transform(Image.fromarray(img_b_3ch, mode='RGB'))
        else:
            img_b_array = np.array(image_b).astype('float').copy()
            img_b_norm = img_b_array / img_b_array.max()
            item_B = self.transform(Image.fromarray(img_b_norm))
        
        # select_filenames
        fn_a = self.files_A_orig_names[index % len(self.files_A_orig_names)]
        fn_b = self.files_B_orig_names[index % len(self.files_B_orig_names)]
        fn_a_c = self.files_A_conv_names[index % len(self.files_A_conv_names)]
        fn_b_c = self.files_B_conv_names[index % len(self.files_B_conv_names)]
        
        return {'A': item_A, 'B': item_B, 'A_fn': fn_a, 'B_fn': fn_b, 'Ac_fn': fn_a_c, 'Bc_fn': fn_b_c}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))