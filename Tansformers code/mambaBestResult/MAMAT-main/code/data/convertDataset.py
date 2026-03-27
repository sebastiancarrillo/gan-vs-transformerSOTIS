import os
import cv2

# Rutas de tu dataset original y carpeta de salida
DATASET_ROOT = "/home/sebastian/sebas/TMT-Sebas/datasetStatic/syn_static"  # Carpeta donde están train_turb/ y test_turb/
OUTPUT_ROOT = "/media/Data/SebasDatasets/SotisForMamba"

splits = {"train_turb": "train", "test_turb": "test"}

# Crear estructura de carpetas
for split_folder in splits.values():
    os.makedirs(os.path.join(OUTPUT_ROOT, split_folder, "gt"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, split_folder, "turb"), exist_ok=True)

for src_split, dst_split in splits.items():
    src_path = os.path.join(DATASET_ROOT, src_split)
    for folder_name in os.listdir(src_path):
        folder_path = os.path.join(src_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Crear carpetas destino
        dst_gt_folder = os.path.join(OUTPUT_ROOT, dst_split, "gt", folder_name)
        dst_turb_folder = os.path.join(OUTPUT_ROOT, dst_split, "turb", folder_name)
        os.makedirs(dst_gt_folder, exist_ok=True)
        os.makedirs(dst_turb_folder, exist_ok=True)

        # Procesar frames turbulentos y crear videos de 10 en 10
        turb_src = os.path.join(folder_path, "turb")
        if os.path.exists(turb_src):
            frames = sorted([f for f in os.listdir(turb_src) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            num_frames = len(frames)
            
            for i in range(0, num_frames, 10):
                block_frames = frames[i:i+10]
                if len(block_frames) < 10:
                    continue  # Saltar si no hay 10 frames completos

                first_frame_name = os.path.splitext(block_frames[0])[0]
                video_path = os.path.join(dst_turb_folder, f"{first_frame_name}.mp4")

                # Leer el primer frame para obtener tamaño
                first_frame_path = os.path.join(turb_src, block_frames[0])
                img = cv2.imread(first_frame_path)
                h, w, _ = img.shape
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

                for f_name in block_frames:
                    frame_path = os.path.join(turb_src, f_name)
                    img = cv2.imread(frame_path)
                    out.write(img)
                
                out.release()
                print(f"Video creado: {video_path}")
        
        # Procesar gt.png y crear video (sin cambios)
        gt_path = os.path.join(folder_path, "gt.png")
        if os.path.exists(gt_path):
            img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            img_rgb = cv2.merge([img, img, img])
            h, w, _ = img_rgb.shape
            video_path = os.path.join(dst_gt_folder, "gt_video.mp4")
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
            for _ in range(10):  # 10 frames
                out.write(img_rgb)
            out.release()
            print(f"Video creado: {video_path}")
        else:
            print(f"No se encontró gt.png en {folder_path}")
