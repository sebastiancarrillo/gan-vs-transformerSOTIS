import sys
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure as tmf_ssim
from torchmetrics.functional import peak_signal_noise_ratio as tmf_psnr
import torchvision.transforms.functional as TF
import cv2

# --- Fix imports ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from model.TMT_DC import TMT_MS
from data.datasetVideoEvaluation import DataLoaderTurbVideo


# =====================================================
# CONFIGURACIÓN
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/media/Data/sebasModels/BestModelsTurbulenceMitg/mambaBestResult/logs/checkpoints/model_best.pth"
VIDEOS_PATH = "/media/Data/sebasModels/BestModelsTurbulenceMitg/zImagestTry/videos"
OUTPUT_PATH = "/media/Data/sebasModels/BestModelsTurbulenceMitg/outputImages/mambaBestResult"
NUM_FRAMES = 10
EXTENDED = 0


# =====================================================
# UTILIDADES
# =====================================================
def calc_ssim(img_pred, img_gt):
    return tmf_ssim(img_pred.clamp(0, 1), img_gt.clamp(0, 1), data_range=1.0).item()


def calc_psnr(img_pred, img_gt):
    return tmf_psnr(img_pred.clamp(0, 1), img_gt.clamp(0, 1), data_range=1.0).item()


def load_model(model_path, device):
    model = TMT_MS(
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=2,
        n_frames=NUM_FRAMES,
        swindef=False,
        mambadef=False,
        dim=16,
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print("✅ Modelo cargado correctamente.")
    return model


def useModel(input_, model):
    with torch.no_grad():
        output = model(input_.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
    return output


def save_first_frame(output_frames, seq_path):
    """Guarda solo el primer frame restaurado"""
    folder_name = os.path.basename(seq_path).replace(".mp4", "")
    out_dir = os.path.join(OUTPUT_PATH, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    img_np = output_frames[0].squeeze().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    out_path = os.path.join(out_dir, "restored_output.png")
    Image.fromarray(img_np).save(out_path)
    print(f"💾 Guardada imagen: {out_path}")


# =====================================================
# MAIN
# =====================================================
def main():
    print("🚀 Cargando modelo...")
    model = load_model(MODEL_PATH, DEVICE)
    print("Modelo cargado ✅")

    print(f"📂 Evaluando videos en: {VIDEOS_PATH}")
    val_dataset = DataLoaderTurbVideo(
        VIDEOS_PATH,
        num_frames=NUM_FRAMES,
        patch_size=256,
        noise=0.0001,
        is_train=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
    )

    print(f"🔍 Total de videos: {len(val_dataset)}")

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            input_ = data[0].to(DEVICE)
            target = data[1].to(DEVICE)
            seq_path = data[2][0]

            print(f"\n▶ Procesando video {idx+1}/{len(val_dataset)}: {seq_path}")

            # Inferencia
            output = useModel(input_, model)
            restored_frames = [output[:, i, ...] for i in range(output.shape[1])]

            # Métricas
            ssim_before = np.mean([calc_ssim(input_[:, i, ...], target[:, i, ...]) for i in range(NUM_FRAMES)])
            ssim_after = np.mean([calc_ssim(restored_frames[i], target[:, i, ...]) for i in range(NUM_FRAMES)])
            psnr_before = np.mean([calc_psnr(input_[:, i, ...], target[:, i, ...]) for i in range(NUM_FRAMES)])
            psnr_after = np.mean([calc_psnr(restored_frames[i], target[:, i, ...]) for i in range(NUM_FRAMES)])

            print(f"   SSIM antes:  {ssim_before:.4f}")
            print(f"   SSIM después: {ssim_after:.4f}")
            print(f"   PSNR antes:  {psnr_before:.2f} dB")
            print(f"   PSNR después: {psnr_after:.2f} dB")

            # Guardar solo el primer frame
            save_first_frame(restored_frames, seq_path)

    print(f"\n✅ Finalizado. Resultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
