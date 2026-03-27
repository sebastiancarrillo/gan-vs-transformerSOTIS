import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from tqdm import tqdm
from math import log10
import argparse
import os
import re
import random
import pandas as pd
import matplotlib.pyplot as plt

from utils.scheduler import GradualWarmupScheduler, CosineDecayWithWarmUpScheduler
import utils.losses as losses
from utils import utils_image as util
from torchmetrics.functional import structural_similarity_index_measure as tmf_ssim
from torchmetrics.functional import peak_signal_noise_ratio as tmf_psnr
from utils.general import create_log_folder, get_cuda_info, find_latest_checkpoint
import torchvision.transforms.functional as TF

# === Ajusta tus rutas ===
from data.datasetEvaluation import DataLoaderTurbImage
from model.TMT import TMT_MS
model_script_path = '/home/sebastian/sebas/TMT-Sebas/logs/increasingStridesExtendedModelNewLr_09-11-2024-16-13-50'
sys.path.append(model_script_path)
from scipts.train_TMT_static import ExtendedModel, inptLayerExtNetwork
# ==========================


def eval_tensor_imgs(gt, output):
    """Calcula PSNR y SSIM promedio por batch"""
    psnr_list, ssim_list = [], []
    for b in range(output.shape[0]):
        psnr_tmp, ssim_tmp = [], []
        for i in range(output.shape[1]):
            img = output[b, i, ...].data.clamp_(0, 1).unsqueeze(0)
            img_gt = gt[b, i, ...].data.clamp_(0, 1).unsqueeze(0)
            psnr_tmp.append(tmf_psnr(img, img_gt, data_range=1.0).item())
            ssim_tmp.append(tmf_ssim(img, img_gt, data_range=1.0).item())
        psnr_list.append(np.mean(psnr_tmp))
        ssim_list.append(np.mean(ssim_tmp))
    return psnr_list, ssim_list


def load_model(model_path, device, extended):
    """Carga el modelo entrenado"""
    model = TMT_MS(num_blocks=[2,3,3,4], num_refinement_blocks=2, n_frames=12, att_type='shuffle').to(device)
    if extended == 1:
        model = ExtendedModel(model).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def useModel(input_, target, model, extended):
    """Ejecuta el modelo TMT o TMT extendido"""
    img = input_[0, 0, ...].clamp(0, 1).unsqueeze(0)
    img_gt = target[0, 0, ...].clamp(0, 1).unsqueeze(0)
    if extended == 1:
        inputExtLayer = inptLayerExtNetwork(img, img_gt)
        output = model(input_.permute(0, 2, 1, 3, 4), inputExtLayer).permute(0, 2, 1, 3, 4)
    else:
        output = model(input_.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
    return output


def find_best10(model, device, extended):
    """Evalúa todo el dataset y obtiene las 10 imágenes con mayor mejora en SSIM"""
    from tqdm import tqdm
    args = argparse.Namespace(
        num_frames=12,
        patch_size=120,
        num_workers=8,
        batch_size=1
    )
    dataset_path = '/home/sebastian/sebas/TMT-Sebas/datasetStatic/syn_static/test_turb'

    best_results = []

    with torch.no_grad():
        for i in tqdm(range(0, 320), desc="Evaluando dataset completo"):
            dataset_eval = DataLoaderTurbImage(
                rgb_dir=dataset_path,
                num_frames=args.num_frames,
                total_frames=50,
                im_size=args.patch_size,
                noise=0.000,
                is_train=False,
                epoch=i
            )
            dataloader_eval = DataLoader(
                dataset_eval,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
                pin_memory=True
            )

            for s, data in enumerate(dataloader_eval):
                input_ = data[1].to(device)
                target = data[2].to(device)
                seq_path = data[3]
                frame_idx = data[4]

                turb_frames = frame_idx['turb']
                turb_numbers = [int(t.item()) for t in turb_frames]

                turb_folder = os.path.join(seq_path[0], 'turb')
                turb_images = sorted([f for f in os.listdir(turb_folder) if f.endswith('.png')])
                if len(turb_images) <= turb_numbers[0]:
                    continue
                first_frame_name = turb_images[turb_numbers[0]]

                # Inference
                output = useModel(input_, target, model, extended)

                # Métricas
                psnr_rec, ssim_rec = eval_tensor_imgs(target, output)
                psnr_turb, ssim_turb = eval_tensor_imgs(target, input_)

                delta_ssim = np.mean(np.array(ssim_rec) - np.array(ssim_turb))
                best_results.append((delta_ssim, first_frame_name))

    # Ordenar por incremento de SSIM
    best_results = sorted(best_results, key=lambda x: x[0], reverse=True)[:10]
    top10_names = [name for _, name in best_results]

    print("\n=== Top 10 imágenes con mayor incremento de SSIM ===")
    for i, name in enumerate(top10_names, 1):
        print(f"{i}. {name}")

    return top10_names


def main():
    model_path = '/home/sebastian/sebas/TMT-Sebas/logs/mixedTresMetrics_08-02-2024-14-37-24/checkpoints/model_best.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extended = 0  # o 1 si usas modelo extendido

    print("Cargando modelo...")
    model = load_model(model_path, device, extended)
    print("Modelo cargado correctamente ✅")

    best10 = find_best10(model, device, extended)
    print("\nLista final de las 10 mejores imágenes:")
    print(best10)


if __name__ == "__main__":
    main()
