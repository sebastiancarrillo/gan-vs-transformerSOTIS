import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model.UNet3d_TMT import DetiltUNet3DS
from utils.general import get_cuda_info
import utils.losses as losses
from data.dataset_video_train import DataLoaderTurbVideo
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Function to display images
def display_images(input_img, output_img, target_img, index):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(input_img)
    ax[0].set_title('Input')
    ax[0].axis('off')
    
    ax[1].imshow(output_img)
    ax[1].set_title('Output')
    ax[1].axis('off')
    
    ax[2].imshow(target_img)
    ax[2].set_title('Target')
    ax[2].axis('off')
    
    plt.savefig(f'result_{index}.png')
    plt.show()

def show_image(image_tensor, title=''):
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device found? ", device)
    
    model = DetiltUNet3DS(norm='LN', conv_type='dw', residual='pool', noise=0.0001).to(device)
    checkpoint = torch.load('/home/sebastian/sebas/TMT-main/logs/dynamic-tilt_06-11-2024-13-40-07/checkpoints/model_last.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Prepare the test dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # Add other transformations if any
    ])

    #test_dataset = DataLoaderTurbVideo(root='/home/sebastian/sebas/TMT-main/datasetDynamic/turb_syn_videos/train', num_frames=12, patch_size=140, is_train=False)
    #test_dataset = DataLoaderTurbVideo(
    #dataset_dir='/home/sebastian/sebas/TMT-main/datasetDynamic/turb_syn_videos/train', num_frames=12, patch_size=140, train=False)
    #test_dataset =DataLoaderTurbVideo('/home/sebastian/sebas/TMT-main/datasetDynamic/turb_syn_videos/train', num_frames=12, patch_size=13, noise=0.0001, is_train=True)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_dataset = DataLoaderTurbVideo('/home/sebastian/sebas/TMT-main/datasetDynamic/turb_syn_videos/train', num_frames=12, patch_size=140, is_train=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)


    # Show image seb
    data_iter = iter(test_loader)
    sample_batch = next(data_iter)
    sample_input = sample_batch[1][0, 0, ...]
    sample_target = sample_batch[0][0, 0, ...]
    
    show_image(sample_input, title='Sample Input Image')
    show_image(sample_target, title='Sample Target Image')
    #end show image seb

    criterion_char = losses.CharbonnierLoss()

    # Run inference on 10 images
    for idx, data in enumerate(test_loader):
        if idx >= 10:
            break
        
        input_ = data[1].to(device)
        target = data[0].to(device)
        print(f"input size: {input_.size()}")
        print(f"target size: {target.size()}")

        with torch.no_grad():
            output_3, output_2, output = model(input_)
            print(f"output_3 size: {output_3.size()}")
            print(f"output_2 size: {output_2.size()}")
            print(f"output size: {output.size()}")
            loss = 0.6 * criterion_char(output, target) + \
                    0.3 * criterion_char(output_2, target) + \
                    0.1 * criterion_char(output_3, target)
        
        input_img = input_[0, 0, ...].cpu().numpy().transpose(1, 2, 0) * 255.0
        output_img = output[0, 0, ...].cpu().numpy().transpose(1, 2, 0) * 255.0
        target_img = target[0, 0, ...].cpu().numpy().transpose(1, 2, 0) * 255.0

        input_img = input_img.astype(np.uint8)
        output_img = output_img.astype(np.uint8)
        target_img = target_img.astype(np.uint8)

        display_images(input_img, output_img, target_img, idx)

if __name__ == '__main__':
    main()
