import sys
import os
import torch
import numpy as np
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure as tmf_ssim
import utils_image as util

# === Import model and utilities ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.TMT import TMT_MS
model_script_path = os.path.abspath('../addingMeanImageToInput_12-26-2024-17-02-36')
sys.path.append(model_script_path)
# ==================================

# === CONFIGURATION ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '/media/Data/sebasModels/BestModelsTurbulenceMitg/3addingMeanTMT/addingMeanImageToInput_12-26-2024-17-02-36/checkpoints/model_best.pth'
FOLDER_PATH = '/media/Data/sebasModels/BestModelsTurbulenceMitg/zImagestTry'
OUTPUT_ROOT = '/media/Data/sebasModels/BestModelsTurbulenceMitg/outputImages/3addingMeanTMT'
EXTENDED = 0
NUM_FRAMES = 13   # 12 frames + 1 average frame (added by dataloader)
# =====================


def load_model(model_path, device, extended):
    """Load the trained TMT model"""
    model = TMT_MS(num_blocks=[2,3,3,4],
                   num_refinement_blocks=2,
                   n_frames=NUM_FRAMES,
                   att_type='shuffle').to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def calc_ssim(img_pred, img_gt):
    """Compute SSIM between two 4D tensors [1,1,H,W]"""
    return tmf_ssim(img_pred.clamp(0,1), img_gt.clamp(0,1), data_range=1.0).item()


def load_image_tensor(img_path):
    """Load grayscale image as tensor [1,1,H,W]"""
    img = util.imread_uint(img_path, 1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.squeeze(img, -1)
    img = util.uint2single(img)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    return tensor


def useModel(input_, target, model, extended):
    """Forward pass through TMT"""
    if extended == 1:
        from scipts.train_TMT_static import inptLayerExtNetwork
        inputExtLayer = inptLayerExtNetwork(target, target)
        output = model(input_.permute(0,2,1,3,4), inputExtLayer).permute(0,2,1,3,4)
    else:
        output = model(input_.permute(0,2,1,3,4)).permute(0,2,1,3,4)
    return output


def save_output_image(output_frames, seq_folder):
    """Save average restored frame as PNG in OUTPUT_ROOT/<folder_name>/restored_output.png"""
    folder_name = os.path.basename(seq_folder.rstrip("/"))
    output_dir = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    output_tensor = torch.mean(torch.stack(output_frames, dim=0), dim=0)  # [1,1,H,W]
    img_np = output_tensor.squeeze().cpu().clamp(0, 1).numpy()
    img_np = (img_np * 255).astype(np.uint8)

    save_path = os.path.join(output_dir, "restored_output.png")
    Image.fromarray(img_np).save(save_path)
    print(f"   💾 Saved output image: {save_path}")


def process_sequence(seq_folder, model):
    """Evaluate SSIM improvement for one sequence folder and save result image"""
    turb_folder = os.path.join(seq_folder, "frames")
    gt_files = [f for f in os.listdir(seq_folder)
                if f.endswith('.png') or f.endswith('.jpg')]
    if len(gt_files) == 0:
        print(f"⚠️ No groundtruth image found in {seq_folder}")
        return None
    gt_path = os.path.join(seq_folder, gt_files[0])

    turb_images = sorted([os.path.join(turb_folder, f)
                          for f in os.listdir(turb_folder)
                          if f.endswith('.png') or f.endswith('.jpg')])
    if len(turb_images) < 12:
        print(f"⚠️ Only {len(turb_images)} frames found in {turb_folder}, expected at least 12")
        return None
    turb_images = turb_images[:12]

    # === ADD AVERAGE IMAGE (as in the dataloader) ===
    turb_imgs_PIL = [Image.open(p).convert('L') for p in turb_images]          # open as grayscale
    turb_arrays = [np.array(img, dtype=np.float32) for img in turb_imgs_PIL]   # convert to arrays
    avg_array = np.mean(turb_arrays, axis=0).astype(np.uint8)                  # pixel-wise mean
    avg_image = Image.fromarray(avg_array)                                     # back to PIL image
    # Save temporarily to use the same loading path-based logic
    avg_path = os.path.join(seq_folder, "_temp_avg.png")
    avg_image.save(avg_path)
    turb_images.append(avg_path)
    # === END AVERAGE ADDITION ===

    # --- Load tensors ---
    imgs = [load_image_tensor(p).to(DEVICE) for p in turb_images]
    input_batch = torch.stack(imgs, dim=1)  # [1,13,1,H,W]
    gt_img = load_image_tensor(gt_path).to(DEVICE)

    # --- SSIM before (exclude avg frame) ---
    ssim_before = np.mean([calc_ssim(x, gt_img) for x in imgs[:-1]])

    # --- Forward ---
    with torch.no_grad():
        output = useModel(input_batch, gt_img.unsqueeze(1).repeat(1, NUM_FRAMES, 1, 1, 1),
                          model, EXTENDED)
    restored_frames = [output[:, i, ...] for i in range(output.shape[1])]
    ssim_after = np.mean([calc_ssim(r, gt_img) for r in restored_frames])

    # --- Save output image ---
    save_output_image(restored_frames, seq_folder)

    # --- Remove temporary average image ---
    if os.path.exists(avg_path):
        os.remove(avg_path)

    return ssim_before, ssim_after


def main():
    print("Cargando modelo...")
    model = load_model(MODEL_PATH, DEVICE, EXTENDED)
    print("Modelo cargado ✅")

    subfolders = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH)
                  if os.path.isdir(os.path.join(FOLDER_PATH, f))]
    if len(subfolders) == 0:
        print("⚠️ No subfolders found inside:", FOLDER_PATH)
        return

    print(f"\n🔍 Found {len(subfolders)} sequences to evaluate.\n")

    for seq_folder in subfolders:
        print(f"📂 Evaluating: {seq_folder}")
        result = process_sequence(seq_folder, model)
        if result is None:
            continue
        ssim_before, ssim_after = result
        print(f"   SSIM antes:   {ssim_before:.4f}")
        print(f"   SSIM después: {ssim_after:.4f}")
        print(f"   Mejora ΔSSIM: {ssim_after - ssim_before:.4f}\n")

    print(f"\n✅ Finished evaluation. Outputs saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
