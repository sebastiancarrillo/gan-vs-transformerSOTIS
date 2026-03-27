from logging import root
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


     
class DataLoaderTurbImage(Dataset):
    # equal to previous dynamic
    def __init__(self, rgb_dir, num_frames=12, total_frames=60, im_size=None, noise=None, other_turb=None, is_train=True):
        super(DataLoaderTurbImage, self).__init__()
        #esto va a recibir la direccion de sotis en donde se encuentran las carpetas de deblurred, distorted(turb) and groundtruth stabilized(blur)
        ##dejar esto original y mover imagenes manualmente a la carpeta de static para hacer un entrenamiento si funciona, copiar mas imagenes o todas
        #entender como funciona la primera red, cuantas imagenes recibe
        self.num_frames = num_frames
        self.img_list = [os.path.join(rgb_dir, d) for d in os.listdir(rgb_dir)]
        self.img_list = [d for d in self.img_list if len(os.listdir(d))>=3]
        #self.img_list = [os.path.join(rgb_dir, 'Groundtruth/EurasianCitiesGT')]
        #print(rgb_dir)
        #print("el anterioooooooooooooo")
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
        #print("el anterioooooooooooooo")
        #ps = self.ps
        #blur_img_list = [os.path.join(seq_path, 'blur', '{:d}.jpg'.format(n)) for n in frame_idx]
        #if self.other_turb:
        #    turb_img_list = [os.path.join(self.other_turb, 'turb_out', '{:d}.jpg'.format(n)) for n in frame_idx]
        #else:
        #    turb_img_list = [os.path.join(seq_path, 'turb', '{:d}.jpgdddd'.format(n)) for n in frame_idx]
        blur_files = sorted(os.listdir(os.path.join(seq_path, 'blur')))
        #random.shuffle(blur_files)
        if self.other_turb:
            turb_files = sorted(os.listdir(os.path.join(self.other_turb, 'turb_out')))
            
        else:
            turb_files = sorted(os.listdir(os.path.join(seq_path, 'turb')))
        #random.shuffle(turb_files)

        #print("frame_idx", frame_idx)
        frame_idxBlur =  frame_idx.get('blur')
        frame_idxTurb =  frame_idx.get('turb')

        ps = self.ps
        blur_img_list = [os.path.join(seq_path, 'blur', blur_files[i]) for i in frame_idxBlur]
        turb_img_list = [os.path.join(seq_path, 'turb', turb_files[i]) for i in frame_idxTurb]


        blur_imgs = [Image.open(p) for p in blur_img_list]
        turb_imgs = [Image.open(p) for p in turb_img_list]
        #tar_img = Image.open(os.path.join(seq_path, 'gt.png'))
        tar_img = Image.open(os.path.join(seq_path, 'gt.png')).convert('L')
        #print(f"tar_img size (val): {tar_img.size}")
        w,h = tar_img.size
        #print("w:,", w," t: ",t)
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
            noise_level = self.noise * random.random()
            blur_imgs = [self._inject_noise(img, noise_level) for img in blur_imgs]
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
        return blur_imgs, turb_imgs, [tar_img]*self.num_frames
                         
    def _fetch_chunk_train(self, seq_path, frame_idx):
        blur_files = sorted(os.listdir(os.path.join(seq_path, 'blur')))
        #random.shuffle(blur_files)
        if self.other_turb:
            turb_files = sorted(os.listdir(os.path.join(self.other_turb, 'turb_out')))
            
        else:
            turb_files = sorted(os.listdir(os.path.join(seq_path, 'turb')))
        #random.shuffle(turb_files)

        #print("frame_idx", frame_idx)
        frame_idxBlur =  frame_idx.get('blur')
        frame_idxTurb =  frame_idx.get('turb')

        ps = self.ps
        blur_img_list = [os.path.join(seq_path, 'blur', blur_files[i]) for i in frame_idxBlur]
        turb_img_list = [os.path.join(seq_path, 'turb', turb_files[i]) for i in frame_idxTurb]

        blur_imgs = [Image.open(p) for p in blur_img_list]
        turb_imgs = [Image.open(p) for p in turb_img_list]
        #tar_img = Image.open(os.path.join(seq_path, 'gt.png'))
        tar_img = Image.open(os.path.join(seq_path, 'gt.png')).convert('L')
        w,h = tar_img.size
        #print(f"tar_img size (train): {tar_img.size}")
        #print(f"w,t (train): {w,h}")

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

        #print("tamaño del target en tensor: ",tar_img.shape) 

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
        return blur_imgs, turb_imgs, [tar_img]*self.num_frames
                           
    def __getitem__(self, index):
        #since blur has no many images im gonna make it random the images to take for the num_frames used
        #in the other hand, turb has more images and there are groups of 50(grouped by constant c2),
        # so im taking n frames starting form a rand int that determines the block(grou of 50)
        # and another rand it to determine a number from 0 to 50 to fix the start index
        #para asegurarse de que index esta entre 0 y sizeX(cantidadde imagenes groundtruth)
        index_ = index % self.sizex

        img_name = self.img_list[index]

        self.total_frames_blur = len([f for f in os.listdir(os.path.join(img_name, 'blur')) if f.endswith('.png')])
        self.total_frames_turb = len([f for f in os.listdir(os.path.join(img_name, 'turb')) if f.endswith('.png')])

        
        #blur
        start_frame_id = random.randint(0, self.total_frames_blur-self.num_frames)
        frame_idxB = [i for i in range(start_frame_id, start_frame_id+self.num_frames)]
        
        #este random va de 0 al maximo de imagenes que tiene la carpeta - el num de frames
        start_frame_id = random.randint(0, self.total_frames_turb-self.num_frames)

        #turb
        #se resta uno porque los bloques de 50 empiezan a conar desdde el 0 entonces debo eliminar el ultimo qe estaria out of index
        #train making sure the frames have similar level of turbulcnce and blur
        BlockC2 = random.randint(0,-1+(self.total_frames_turb/50) )
        startOnBlock = random.randint(0, 50-self.num_frames)
        start_frame_id=BlockC2*50+startOnBlock
        frame_idxT = [i for i in range(start_frame_id, start_frame_id+self.num_frames)]
        #to train whitout taking into account turbulence and blur similarity for frames use
        #frame_idxT = random.sample(range(self.total_frames_turb), self.num_frames)
        frame_idx = {'blur': frame_idxB, 'turb': frame_idxT}


        seq_path = self.img_list[index_]



        if self.train:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_train(seq_path, frame_idx)
        else:
            blur_imgs, turb_imgs, tar_imgs = self._fetch_chunk_val(seq_path, frame_idx)
        return torch.stack(blur_imgs, dim=0), torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0)
        
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



