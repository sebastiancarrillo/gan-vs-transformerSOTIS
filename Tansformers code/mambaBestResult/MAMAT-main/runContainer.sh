#!/bin/bash
# ======================================================
# Run training/evaluation container for MAMAT (Turbulence project)
# ======================================================

PROJECT_DIR="$(pwd)"
TRAIN_DIR="/media/Data/SebasDatasets/SotisForMamba/train"
VAL_DIR="/media/Data/SebasDatasets/SotisForMamba/test"
LOG_DIR="/media/Data/sebasModels/BestModelsTurbulenceMitg/mambaBestResult/logs"
ZIMAGES_DIR="/media/Data/sebasModels/BestModelsTurbulenceMitg/zImagestTry"
OUTPUT_DIR="/media/Data/sebasModels/BestModelsTurbulenceMitg/outputImages"
IMAGE_NAME="turb_mit"
LOG_DIR_ALT="/media/Data/sebasModels/logsMambaaaaa"
mkdir -p "${LOG_DIR_ALT}"

# Make sure necessary folders exist on the host
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ---- Launch container ----
sudo docker run --gpus all -it --ipc=host --rm \
  --workdir /app/code \
  -v "${PROJECT_DIR}:/app/code" \
  -v "${TRAIN_DIR}:${TRAIN_DIR}:ro" \
  -v "${VAL_DIR}:${VAL_DIR}:ro" \
  -v "${LOG_DIR}:${LOG_DIR}" \
  -v "${LOG_DIR_ALT}:${LOG_DIR_ALT}" \
  -v "${ZIMAGES_DIR}:${ZIMAGES_DIR}:ro" \
  -v "${OUTPUT_DIR}:${OUTPUT_DIR}" \
  -e LOG_PATH="${LOG_DIR}" \
  "${IMAGE_NAME}" \
  bash
