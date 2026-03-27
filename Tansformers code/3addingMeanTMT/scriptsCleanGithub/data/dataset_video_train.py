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
TMT Dataset Loader
-------------------
This module defines dataset classes for training and testing the Turbulence Mitigation Transformer (TMT) model.
It supports loading multi-frame sequences of distorted (turbulent) and blurred images, aligned with corresponding ground truth frames.
Includes preprocessing such as cropping, padding, noise injection, and optional data augmentation.

Classes:
- DataLoaderTurbImage: Loads training and validation samples from structured turbulence/blur/ground-truth folders.
- DataLoaderTurbImageTest: Loads evaluation samples from turbulence folders only.
- DataLoaderBlurImageTest: Loads evaluation samples from external blur directories.

Used in conjunction with the TMT model for video restoration under atmospheric distortion.
"""

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

     
class DataLoaderTurbImage(Dataset):
    # equal to previous dynamic
    def __init__(self, rgb_dir, num_frames=12, total_frames=60, im_size=None, noise=None, other_turb=None, is_train=True):
        super(DataLoaderTurbImage, self).__init__()
        
        self.num_frames = num_frames
        self.img_list = [os.path.join(rgb_dir, d) for d in os.listdir(rgb_dir)]
        self.img_list = [d for d in self.img_list if len(os.listdir(d))>=3]
        self.total_frames = total_frames
        self.ps = im_size
        self.sizex = len(self.img_list)  # get the size of target
        self.train = is_train
        self.noise = noise
        self.other_turb = other_turb
         # Initialize total_frames_blur and total_frames_turb here or within your methods
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

        #get frame indx
        frame_idxBlur =  frame_idx.get('blur')
        frame_idxTurb =  frame_idx.get('turb')

        ps = self.ps
        blur_img_list = [os.path.join(seq_path, 'blur', blur_files[i]) for i in frame_idxBlur]
        turb_img_list = [os.path.join(seq_path, 'turb', turb_files[i]) for i in frame_idxTurb]

        #adding average image at the end
        blur_imgs = [Image.open(p) for p in blur_img_list]
        turb_imgs = [Image.open(p) for p in turb_img_list]

        # Convertir imágenes a arrays de numpy para calcular el promedio
        blur_arrays = [np.array(img, dtype=np.float32) for img in blur_imgs]
        turb_arrays = [np.array(img, dtype=np.float32) for img in turb_imgs]
        # Calcular el promedio pixel a pixel
        blur_avg_array = np.mean(blur_arrays, axis=0).astype(np.uint8)
        turb_avg_array = np.mean(turb_arrays, axis=0).astype(np.uint8)

        # Convertir los arrays de promedio de vuelta a imágenes
        blur_avg_image = Image.fromarray(blur_avg_array)
        turb_avg_image = Image.fromarray(turb_avg_array)

        # Agregar las imágenes promedio al final de los arreglos
        blur_imgs.append(blur_avg_image)
        turb_imgs.append(turb_avg_image)
        ### ading average image END

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
            print('noise is being added')
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
        return blur_imgs, turb_imgs, [tar_img]*(self.num_frames+1)
                         
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
        #add the mean image at the end of the array
        # Convertir imágenes a arrays de numpy para calcular el promedio
        blur_arrays = [np.array(img, dtype=np.float32) for img in blur_imgs]
        turb_arrays = [np.array(img, dtype=np.float32) for img in turb_imgs]
        # Calcular el promedio pixel a pixel
        blur_avg_array = np.mean(blur_arrays, axis=0).astype(np.uint8)
        turb_avg_array = np.mean(turb_arrays, axis=0).astype(np.uint8)

        # Convertir los arrays de promedio de vuelta a imágenes
        blur_avg_image = Image.fromarray(blur_avg_array)
        turb_avg_image = Image.fromarray(turb_avg_array)

        # Agregar las imágenes promedio al final de los arreglos
        blur_imgs.append(blur_avg_image)
        turb_imgs.append(turb_avg_image)

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
        Selects a random sequence of frames from both the 'blur' and 'turb' folders for training or validation.
        
        Blur sequences are grouped in blocks of 20 frames, while turbulence sequences are grouped in blocks of 50.
        The function randomly selects a block and starting index for each input type, builds frame indices, 
        and returns stacked tensors of the corresponding image sequences along with the sequence path and indices.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[Tensor, Tensor, Tensor, str, dict]: 
                - blur sequence (Tensor)
                - turbulence sequence (Tensor)
                - target sequence (Tensor)
                - sequence folder path
                - dictionary of selected frame indices
        """
        index_ = index % self.sizex
        

        img_name = self.img_list[index]
        # Count the number of available frames for blur and turbulence
        self.total_frames_blur = len([f for f in os.listdir(os.path.join(img_name, 'blur')) if f.endswith('.png')])
        self.total_frames_turb = len([f for f in os.listdir(os.path.join(img_name, 'turb')) if f.endswith('.png')])

        
        # ---- Blur block selection ----
        # Blur frames are grouped in blocks of 20
        BlockB = random.randint(0, -1 + (self.total_frames_blur // 20))
        startOnBlock = random.randint(0, 20 - self.num_frames)
        start_frame_id = BlockB * 20 + startOnBlock
        frame_idxB = [i for i in range(start_frame_id, start_frame_id + self.num_frames)]
        
        
        # ---- Turbulence block selection ----
        # Turbulence frames are grouped in blocks of 50
        BlockC2 = random.randint(0,-1+(self.total_frames_turb/50) )
        startOnBlock = random.randint(0, 50-self.num_frames)
        start_frame_id=BlockC2*50+startOnBlock
        frame_idxT = [i for i in range(start_frame_id, start_frame_id+self.num_frames)]
        frame_idx = {'blur': frame_idxB, 'turb': frame_idxT}


        # ---- Validation indexing (fixed for reproducibility) ----
        random.seed(index_)
        BlockC2 = random.randint(0,-1+(self.total_frames_turb/50) )
        startOnBlock = random.randint(0, 50-self.num_frames)
        start_frame_id=BlockC2*50+startOnBlock
        frame_idxT = [i for i in range(start_frame_id, start_frame_id+self.num_frames)]
        frame_idxVal = {'blur': frame_idxB, 'turb': frame_idxT}
        
        random.seed()#this generates a seed based o the current system time

        seq_path = self.img_list[index]

        if self.train:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_train(seq_path, frame_idx)
            return torch.stack(blur_imgs, dim=0), torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0), seq_path, frame_idx
        else:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_val(seq_path, frame_idxVal)
            return torch.stack(blur_imgs, dim=0), torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0), seq_path, frame_idx
        
        
    """
    def __getitem__(self, index):
        train = self.train
        num_frames = self.num_frames

        img_name = self.img_list[index]
        total_frames = len([f for f in os.listdir(img_name) if f.endswith('.jpg')])
        frame_idx = random.sample(range(total_frames), num_frames)
        frame_idx.sort()

        if train:
            blur_imgs, turb_imgs, tar_img = self._fetch_chunk_train(img_name, frame_idx)
        else:
            blur_imgs, turb_imgs, tar_img = self._fetch_chunk_val(img_name, frame_idx)

        return torch.stack(blur_imgs, 0), torch.stack(turb_imgs, 0), torch.stack(tar_img, 0)
        """
      
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
        index_ = index % self.sizex
        frame_idx = [i for i in range(self.start_frame, self.start_frame + self.num_frames)]
        seq_path = self.img_list[index_]
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
        index_ = index % self.sizex
        frame_idx = [i for i in range(self.num_frames)]
        seq_path = self.img_list[index_]
        turb_imgs, tar_imgs = self._fetch_chunk(seq_path, frame_idx)
        return torch.stack(turb_imgs, dim=0), tar_imgs, seq_path



