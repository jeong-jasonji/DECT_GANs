import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import pydicom
from pydicom import dcmread
import numpy as np
from numpy import asarray
from numpy import expand_dims
import torch
import cv2

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))
        
        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
    
    ### Modified source code  
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        #A_dcm = pydicom.dcmread(A_path)
        #A_pixel = A_dcm.pixel_array
        A_pixel = np.load(A_path)
        
        #downscale for progressive higher res training (32x32 to 512x512)
        A_pixel = cv2.resize(A_pixel, dsize=(64, 64), interpolation=cv2.INTER_LANCZOS4)

        #expand dimension channel last
        A_pixel = expand_dims(A_pixel, axis=2)
        #convert to 3-channel
        A = np.concatenate((A_pixel,)*3, axis=-1)
        #convert to float 32
        A = np.float32(A) 
        
        #linear transformation of raw pixel values to HU values
        A = A*1 + -1024
        
        
        #def min and max pixel values (normalize will occure via transofrm function below)
        A[A>1500]=1500
        A[A<-500]=-500

        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            #scale to -1,1 using above min/max bounds
            max = 1500
            min = -500
            scaled_A = (2*((A-min)/(max-min)))-1.0
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(scaled_A)
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            #B_dcm = pydicom.dcmread(B_path)
            #B_pixel = B_dcm.pixel_array
            B_pixel = np.load(B_path)

	    #downscale for progressive higher res training (32x32 to 512x512)
            B_pixel = cv2.resize(B_pixel, dsize=(64, 64), interpolation=cv2.INTER_LANCZOS4)

            #expand dimension channel last
            B_pixel = expand_dims(B_pixel, axis=2)
            #convert to 3-channel
            B = np.concatenate((B_pixel,)*3, axis=-1)
            #convert to float 32
            B = np.float32(B) 

            #linear transformation of raw pixel values to HU/iodine values
            #slope 1, intercept 0
            B = B*1 + 0

            #def min and max pixel values 
            B[B>250]=250
            B[B<-150]=-150

            #scale to -1,1 using above min/max bounds
            max = 250
            min = -150
            scaled_B = (2*((B-min)/(max-min)))-1.0
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(scaled_B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
