import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as tmf_ssim

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
        
class EdgeLoss3D(nn.Module):
    def __init__(self):
        super(EdgeLoss3D, self).__init__()
        self.el2d = EdgeLoss()

    def forward(self, x, y):
        losses = [self.el2d(x[:,i,...].squeeze(1), y[:,i,...].squeeze(1)) for i in range(x.shape[1])]
        return sum(losses)/len(losses)
        
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class SsimLoss(nn.Module):
    def __init__(self):
        super(SsimLoss, self).__init__()

    def forward(self, prediction, target):
        img = prediction[0, 0, ...].clamp(0, 1).unsqueeze(0)
        img_gt = target[0, 0, ...].clamp(0, 1).unsqueeze(0)
        ssim_loss = 1 - tmf_ssim(img, img_gt, data_range=1.0)
        return ssim_loss
