import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class DataLoaderTurbVideo(Dataset):
    """Data loader for turbulent video mitigation

    Args:
        Dataset (Abstract dataloader): torch.utils.data dataloader base class
    """

    def __init__(
        self, root_dir, num_frames=10, patch_size=None, noise=None, is_train=True
    ):
        super(DataLoaderTurbVideo, self).__init__()
        self.num_frames = num_frames
        self.turb_list = []
        self.blur_list = []
        self.gt_list = []
        
        #v contains folders with numbers like 023-2 inside that should be 
        for v in os.listdir(os.path.join(root_dir, "turb")):
            for video in os.listdir(os.path.join(root_dir, "turb",v)):
                self.gt_list.append(os.path.join(root_dir, "gt", v,'gt_video.mp4'))
                self.turb_list.append(os.path.join(root_dir, "turb",v, video))

            


        self.ps = patch_size
        self.sizex = len(self.gt_list)  # get the size of target
        #print("self.ps",self.ps)
        print("total amount of videos",len(self.gt_list))
        self.train = is_train
        self.noise = noise

    def __len__(self):
        return self.sizex

    def _inject_noise(self, img, noise):
        noise = (noise**0.5) * torch.randn(img.shape)
        out = img + noise
        return out.clamp(0, 1)

    def _fetch_chunk_val(self, idx):
        ps = self.ps
        turb_vid = cv2.VideoCapture(self.turb_list[idx])
        #print("self.turb_list[idx]",self.turb_list[idx])
        seq_path = self.turb_list[idx]
        gt_vid = cv2.VideoCapture(self.gt_list[idx])
        h, w = int(gt_vid.get(4)), int(gt_vid.get(3))
        total_frames = int(gt_vid.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.num_frames:
            print("no enough frame in video " + self.gt_list[idx])
        start_frame_id = (total_frames - self.num_frames) // 2

        # load frames from video
        gt_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        turb_vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        tar_imgs = [gt_vid.read()[1] for i in range(self.num_frames)]
        turb_imgs = [turb_vid.read()[1] for i in range(self.num_frames)]
        turb_vid.release()
        gt_vid.release()

        tar_imgs = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in tar_imgs
        ]
        turb_imgs = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in turb_imgs
        ]

        if ps > 0:
            padw = ps - w if w < ps else 0
            padh = ps - h if h < ps else 0
            if padw != 0 or padh != 0:
                turb_imgs = [
                    TF.pad(img, (0, 0, padw, padh), padding_mode="reflect")
                    for img in turb_imgs
                ]
                tar_imgs = [
                    TF.pad(img, (0, 0, padw, padh), padding_mode="reflect")
                    for img in tar_imgs
                ]

            turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
            tar_imgs = [TF.to_tensor(img) for img in tar_imgs]

            hh, ww = tar_imgs[0].shape[1], tar_imgs[0].shape[2]

            rr = (hh - ps) // 2
            cc = (ww - ps) // 2
            # Crop patch
            turb_imgs = [img[:, rr : rr + ps, cc : cc + ps] for img in turb_imgs]
            tar_imgs = [img[:, rr : rr + ps, cc : cc + ps] for img in tar_imgs]
        else:
            turb_imgs = [TF.to_tensor(img) for img in turb_imgs]
            tar_imgs = [TF.to_tensor(img) for img in tar_imgs]

        if self.noise:
            noise_level = self.noise * random.random()
            turb_imgs = [self._inject_noise(img, noise_level) for img in turb_imgs]
        return turb_imgs, tar_imgs ,seq_path

    
    def __getitem__(self, index):
        index_ = index % self.sizex
        #print('index',index)

        turb_imgs, tar_imgs,seq_path = self._fetch_chunk_val(index_)
        
        return torch.stack(turb_imgs, dim=0), torch.stack(tar_imgs, dim=0),seq_path,index_
