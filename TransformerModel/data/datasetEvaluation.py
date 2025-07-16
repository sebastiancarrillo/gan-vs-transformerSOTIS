from logging import root
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import cv2
import re
"""
TMT Dataset Module
------------------
This file defines custom PyTorch Dataset classes for loading video sequences affected by atmospheric turbulence,
used in training and evaluating the Turbulence Mitigation Transformer (TMT) model.

Each dataset sample contains a sequence of distorted frames (blurred and/or turbulent) and the corresponding
ground truth frame. The structure supports training, validation, and testing across multiple frame configurations.

Classes:
--------
- DataLoaderTurbImage:
    For training and validation with sequences that include:
    - 'blur': stabilized or motion-blurred frames
    - 'turb': turbulence-distorted frames
    - 'gt.png': corresponding ground truth

- DataLoaderTurbImageTest:
    For testing models using only turbulent frames and ground truth ('gt.jpg').

- DataLoaderBlurImageTest:
    For testing on pre-generated blurred outputs (e.g., from other methods or folders).

Key Features:
-------------
- Dynamic frame block selection and cropping
- Noise injection and data augmentation (training only)
- Temporal sequence extraction from organized folders
- Compatible with 5D input format: [batch, sequence_length, channels, height, width]
"""




def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


     
class DataLoaderTurbImage(Dataset):
    # equal to previous dynamic
    def __init__(self, rgb_dir, num_frames=12, total_frames=60, im_size=None, noise=None, other_turb=None, is_train=True,epoch=1000):
        super(DataLoaderTurbImage, self).__init__()
        # Initialize dataset configuration
        # The root directory (rgb_dir) is expected to contain subfolders with:
        # - 'blur': motion-blurred or stabilized images
        # - 'turb': distorted images (with atmospheric turbulence)
        # - 'gt.png': corresponding ground truth image
        
        self.num_frames = num_frames
        # Build list of sequence directories with at least 3 files (to avoid empty or incomplete folders)
        self.img_list = [os.path.join(rgb_dir, d) for d in os.listdir(rgb_dir)]
        self.img_list = [d for d in self.img_list if len(os.listdir(d))>=3]


        self.total_frames = total_frames  # Max frames to load per sequence
        self.ps = im_size                # Patch size for cropping
        self.sizex = len(self.img_list) # Total number of valid sequences
        self.train = is_train           # Whether to use training or validation mode
        self.noise = noise              # Optional noise level to add to input frames
        self.other_turb = other_turb    # Optional external path for turbulence frames
        self.epoch = epoch              # Current epoch index (used for seed control)


         # Initialize total_frames_blur and total_frames_turb here or within your methods
        # Frame counters (populated later in __getitem__)
        self.total_frames_blur = 0
        self.total_frames_turb = 0


    def __len__(self):
        return self.sizex

    def _inject_noise(self, img, noise_level):
        noise = (noise_level**0.5)*torch.randn(img.shape)
        out = img + noise
        return out.clamp(0,1)
        
    def _fetch_chunk_val(self, seq_path, frame_idx):
        blur_files = sorted(os.listdir(os.path.join(seq_path, 'blur')), key=lambda x: float(re.search(r'Cn2=(\d+e[-+]?\d+)', x).group(1)))
        
        if self.other_turb:
            turb_files = sorted(os.listdir(os.path.join(self.other_turb, 'turb_out')))
            
        else:
            turb_files = sorted(os.listdir(os.path.join(seq_path, 'turb')))
        

    
        frame_idxBlur =  frame_idx.get('blur')
        frame_idxTurb =  frame_idx.get('turb')

        ps = self.ps
        blur_img_list = [os.path.join(seq_path, 'blur', blur_files[i]) for i in frame_idxBlur]
        turb_img_list = [os.path.join(seq_path, 'turb', turb_files[i]) for i in frame_idxTurb]

        blur_imgs = [Image.open(p) for p in blur_img_list]
        turb_imgs = [Image.open(p) for p in turb_img_list]

        
        tar_img = Image.open(os.path.join(seq_path, 'gt.png')).convert('L')
        
        w,h = tar_img.size
    
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0
        if padw!=0 or padh!=0:
            blur_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in blur_imgs]
            turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]   
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
                      
        blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]
        
        rr     = (hh-ps) // 2
        cc     = (ww-ps) // 2
         # Crop patch
        blur_imgs = [img[:, rr:rr+ps, cc:cc+ps] for img in blur_imgs]
        turb_imgs = [img[:, rr:rr+ps, cc:cc+ps] for img in turb_imgs]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]
    
        if self.noise:
            print('adding noise!!!!!!!!!!!!!!! to val')
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
        return blur_imgs, turb_imgs, [tar_img]*(self.num_frames)
                         
    def _fetch_chunk_train(self, seq_path, frame_idx):
        blur_files = sorted(os.listdir(os.path.join(seq_path, 'blur')), key=lambda x: float(re.search(r'Cn2=(\d+e[-+]?\d+)', x).group(1)))

        if self.other_turb:
            turb_files = sorted(os.listdir(os.path.join(self.other_turb, 'turb_out')))
            
        else:
            turb_files = sorted(os.listdir(os.path.join(seq_path, 'turb')))
        

        frame_idxBlur =  frame_idx.get('blur')
        frame_idxTurb =  frame_idx.get('turb')

        ps = self.ps
        blur_img_list = [os.path.join(seq_path, 'blur', blur_files[i]) for i in frame_idxBlur]
        turb_img_list = [os.path.join(seq_path, 'turb', turb_files[i]) for i in frame_idxTurb]

        blur_imgs = [Image.open(p) for p in blur_img_list]
        turb_imgs = [Image.open(p) for p in turb_img_list]
        

        tar_img = Image.open(os.path.join(seq_path, 'gt.png')).convert('L')
        w,h = tar_img.size


        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0
        if padw!=0 or padh!=0:
            blur_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in blur_imgs]
            turb_imgs = [TF.pad(img, (0,0,padw,padh), padding_mode='reflect') for img in turb_imgs]   
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')
        
        aug    = random.randint(0, 2)
        if aug == 1:
            blur_imgs = [TF.adjust_gamma(img, 1) for img in blur_imgs]
            turb_imgs = [TF.adjust_gamma(img, 1) for img in turb_imgs]
            tar_img = TF.adjust_gamma(tar_img, 1)
            
        aug    = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_imgs = [TF.adjust_saturation(img, sat_factor) for img in blur_imgs]
            turb_imgs = [TF.adjust_saturation(img, sat_factor) for img in turb_imgs]
            tar_img = TF.adjust_saturation(tar_img, sat_factor)
            
        blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_img = TF.to_tensor(tar_img)


        hh, ww = tar_img.shape[1], tar_img.shape[2]

        enlarge_factor = random.choice([0.8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1,2, 1.5, 1.8, 2])
        crop_size = ps * enlarge_factor
        crop_size = min(hh, ww, crop_size)
        hcro = int(crop_size * random.uniform(1.1, 0.9))
        wcro = int(crop_size * random.uniform(1.1, 0.9))
        hcro = min(hcro, hh)
        wcro = min(wcro, ww)
        rr   = random.randint(0, hh-hcro)
        cc   = random.randint(0, ww-wcro)
        
        # Crop patch
        blur_imgs = [TF.resize(img[:, rr:rr+hcro, cc:cc+wcro], (ps, ps)) for img in blur_imgs]
        turb_imgs = [TF.resize(img[:, rr:rr+hcro, cc:cc+wcro], (ps, ps)) for img in turb_imgs]
        tar_img = TF.resize(tar_img[:, rr:rr+hcro, cc:cc+wcro], (ps, ps))

        if self.noise:
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
            
        aug    = random.randint(0, 8)
        # Data Augmentations
        if aug==1:
            blur_imgs = [img.flip(1) for img in blur_imgs]
            turb_imgs = [img.flip(1) for img in turb_imgs]
            tar_img = tar_img.flip(1)
        elif aug==2:
            blur_imgs = [img.flip(2) for img in blur_imgs]
            turb_imgs = [img.flip(2) for img in turb_imgs]
            tar_img = tar_img.flip(2)
        elif aug==3:
            blur_imgs = [torch.rot90(img, dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img, dims=(1,2)) for img in turb_imgs]
            tar_img = torch.rot90(tar_img, dims=(1,2))
        elif aug==4:
            blur_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in blur_imgs]
            turb_imgs = [torch.rot90(img,dims=(1,2), k=2) for img in turb_imgs]
            tar_img = torch.rot90(tar_img, dims=(1,2), k=2)
        elif aug==5:
            blur_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in blur_imgs]
            turb_imgs = [torch.rot90(img,dims=(1,2), k=3) for img in turb_imgs]
            tar_img = torch.rot90(tar_img, dims=(1,2), k=3)
        elif aug==6:
            blur_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img.flip(1), dims=(1,2)) for img in turb_imgs]
            tar_img = torch.rot90(tar_img.flip(1), dims=(1,2))
        elif aug==7:
            blur_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in blur_imgs]
            turb_imgs = [torch.rot90(img.flip(2), dims=(1,2)) for img in turb_imgs]
            tar_img = torch.rot90(tar_img.flip(2), dims=(1,2))
        return blur_imgs, turb_imgs, [tar_img]*(self.num_frames+1)
                           
    def __getitem__(self, index):
        """
        Returns a sample consisting of a sequence of blur frames, turbulence frames, 
        and a corresponding ground truth target. Frame selection depends on whether 
        the dataset is in training or validation mode.

        Blur frames are selected randomly within blocks of 20.
        Turbulent frames are structured in blocks of 50 and accessed deterministically 
        using the current epoch to ensure coverage.

        Args:
            index (int): Index of the sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor, str, dict]:
                - Stacked blur images
                - Stacked turbulence images
                - Stacked ground truth images
                - Sequence path
                - Dictionary of selected frame indices
        """
        # Get the current epoch (used to determine turbulence frame block)
        epoch =self.epoch
        # Get the sequence directory by index
        img_name = self.img_list[index]

        # Count the number of available blur and turbulence frames
        self.total_frames_blur = len([f for f in os.listdir(os.path.join(img_name, 'blur')) if f.endswith('.png')])
        self.total_frames_turb = len([f for f in os.listdir(os.path.join(img_name, 'turb')) if f.endswith('.png')])

        
        # ---- Blur frame selection (randomized within 20-frame blocks) ----
        BlockB = random.randint(0, -1 + (self.total_frames_blur // 20))
        startOnBlock = random.randint(0, 20 - self.num_frames)
        start_frame_id = BlockB * 20 + startOnBlock
        frame_idxB = [i for i in range(start_frame_id, start_frame_id + self.num_frames)]
        
        # ---- Turbulence frame selection (deterministic based on epoch) ----
        # Turbulent frames are grouped in 50-frame blocks
        # Each 50-frame block is divided into 4 sub-blocks (12 frames each, non-overlapping)
        BlockC2 = int(epoch/4)      # Which 50-frame block to use
        subBlock=epoch%4            # Which sub-block within the 50 frames
        startOnBlock = subBlock*12  # Starting frame within the sub-block
        start_frame_id=BlockC2*50+startOnBlock
        frame_idxT = [i for i in range(start_frame_id, start_frame_id+self.num_frames)]
        frame_idx = {'blur': frame_idxB, 'turb': frame_idxT}


        # ---- Validation frame indexing (uses same logic as training) ----
        BlockC2 = int(epoch/4)#every 4 need to change to the next 50 frames
        subBlock=epoch%4   #number in between 0 to 3 (4 elements, 4 possible options from each 50 frames)
        startOnBlock = subBlock*12
        start_frame_id=BlockC2*50+startOnBlock
        frame_idxT = [i for i in range(start_frame_id, start_frame_id+self.num_frames)]
        frame_idxVal = {'blur': frame_idxB, 'turb': frame_idxT}
        
        # Get the path to the current sequence

        seq_path = self.img_list[index]
        # Load and return data
        if self.train:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_train(seq_path, frame_idx)
            return torch.stack(blur_imgs, dim=0), torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0), seq_path, frame_idx
        else:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_val(seq_path, frame_idxVal)
            return torch.stack(blur_imgs, dim=0), torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0), seq_path, frame_idxVal
        
      
class DataLoaderTurbImageTest(Dataset):
    # equal to previous dynamic
    def __init__(self, rgb_dir, num_frames=48, total_frames=50, im_size=None, noise=None, is_train=True, start_frame=0):
        super(DataLoaderTurbImageTest, self).__init__()
        self.num_frames = num_frames
        self.img_list = [os.path.join(rgb_dir, d) for d in os.listdir(rgb_dir)]
        self.img_list = [d for d in self.img_list if len(os.listdir(d))>=3]
        self.total_frames = total_frames
        self.ps = im_size
        self.sizex = len(self.img_list)  # get the size of target
        self.train = is_train
        self.noise = noise
        self.start_frame = start_frame

    def __len__(self):
        return self.sizex
                         
    def _fetch_chunk(self, seq_path, frame_idx):
        turb_img_list = [os.path.join(seq_path, 'turb', '{:d}.jpg'.format(n)) for n in frame_idx]
        turb_imgs = [Image.open(p) for p in turb_img_list]
        tar_img = Image.open(os.path.join(seq_path, 'gt.jpg'))
        turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
        tar_img = TF.to_tensor(tar_img)
        return turb_imgs, tar_img
                           
    def __getitem__(self, index):
        #index_ = index % self.sizex
        frame_idx = [i for i in range(self.start_frame, self.start_frame + self.num_frames)]
        seq_path = self.img_list[index]
        turb_imgs, tar_imgs = self._fetch_chunk(seq_path, frame_idx)
        return torch.stack(turb_imgs, dim=0), tar_imgs, seq_path

class DataLoaderBlurImageTest(Dataset):
    # equal to previous dynamic
    def __init__(self, rgb_dir, blur_dir, num_frames=48, total_frames=48, im_size=None, noise=None, is_train=True):
        super(DataLoaderBlurImageTest, self).__init__()
        self.num_frames = num_frames
        self.img_list = [os.path.join(rgb_dir, d) for d in os.listdir(rgb_dir)]
        self.img_list = [d for d in self.img_list if len(os.listdir(d))>=3]
        self.blur_dir = blur_dir
        self.total_frames = total_frames
        self.ps = im_size
        self.sizex = len(self.img_list)  # get the size of target
        self.train = is_train
        self.noise = noise

    def __len__(self):
        return self.sizex
                         
    def _fetch_chunk(self, seq_path, frame_idx):
        img_name = os.path.split(seq_path)[-1]
        blur_img_list = [os.path.join(self.blur_dir, img_name, 'turb_out', '{:d}.jpg'.format(n)) for n in frame_idx]
        blur_imgs = [Image.open(p) for p in blur_img_list]
        tar_img = Image.open(os.path.join(seq_path, 'gt.png'))
        blur_imgs = [TF.to_tensor(img) for img in blur_imgs]
        tar_img = TF.to_tensor(tar_img)
        return blur_imgs, tar_img
                           
    def __getitem__(self, index):
        frame_idx = [i for i in range(self.num_frames)]
        seq_path = self.img_list[index]
        turb_imgs, tar_imgs = self._fetch_chunk(seq_path, frame_idx)
        return torch.stack(turb_imgs, dim=0), tar_imgs, seq_path



