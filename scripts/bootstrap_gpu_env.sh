#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CREATE_VENV="${CREATE_VENV:-0}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_PROJECT="${INSTALL_PROJECT:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
DOWNLOAD_HF="${DOWNLOAD_HF:-1}"
FORCE_DOWNLOAD_DATA="${FORCE_DOWNLOAD_DATA:-0}"
INCLUDE_DATASET="${INCLUDE_DATASET:-1}"
INCLUDE_LLAMA3="${INCLUDE_LLAMA3:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
UNZIP_DATA="${UNZIP_DATA:-1}"
DATA_OUT_DIR="${DATA_OUT_DIR:-.}"
DATA_ARCHIVE="${DATA_ARCHIVE:-data/onestop_bundle.zip}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
DEFAULT_GDRIVE_ID="1lvPnyMV_AbKo2CdcAy3kvQa8qIJuVSeg"
DEFAULT_GDRIVE_URL="https://drive.google.com/file/d/${DEFAULT_GDRIVE_ID}/view?usp=sharing"
GDRIVE_URL="${GDRIVE_URL:-}"
GDRIVE_ID="${GDRIVE_ID:-}"

if [[ "${CREATE_VENV}" == "1" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r requirements-cu121.txt
fi

"${PYTHON_BIN}" -m pip install -r requirements.txt

if [[ "${INSTALL_PROJECT}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -e .
fi

mkdir -p "$(dirname "${DATA_ARCHIVE}")"

if [[ "${DOWNLOAD_DATA}" == "1" ]]; then
  if [[ -f "${DATA_ARCHIVE}" && "${FORCE_DOWNLOAD_DATA}" != "1" ]]; then
    echo "Using existing data archive: ${DATA_ARCHIVE}"
  elif [[ -n "${GDRIVE_URL}" ]]; then
    if [[ "${GDRIVE_URL}" =~ /d/([^/]+) ]]; then
      "${PYTHON_BIN}" -m gdown "https://drive.google.com/uc?id=${BASH_REMATCH[1]}" -O "${DATA_ARCHIVE}"
    else
      "${PYTHON_BIN}" -m gdown "${GDRIVE_URL}" -O "${DATA_ARCHIVE}"
    fi
  elif [[ -n "${GDRIVE_ID}" ]]; then
    "${PYTHON_BIN}" -m gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "${DATA_ARCHIVE}"
  else
    "${PYTHON_BIN}" -m gdown "https://drive.google.com/uc?id=${DEFAULT_GDRIVE_ID}" -O "${DATA_ARCHIVE}"
  fi
fi

if [[ "${DOWNLOAD_DATA}" == "1" && -f "${DATA_ARCHIVE}" && "${UNZIP_DATA}" == "1" ]]; then
  case "${DATA_ARCHIVE}" in
    *.zip)
      export DATA_ARCHIVE DATA_OUT_DIR
      "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
import zipfile

archive = Path(os.environ["DATA_ARCHIVE"])
out_dir = Path(os.environ["DATA_OUT_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as zf:
    zf.extractall(out_dir)
print(f"Extracted {archive} -> {out_dir}")
PY
      ;;
    *)
      echo "Downloaded ${DATA_ARCHIVE}; skipping extraction because it is not a .zip file."
      ;;
  esac
fi

if [[ "${DOWNLOAD_HF}" == "1" ]]; then
  HF_ARGS=()
  if [[ -n "${HF_CACHE_DIR}" ]]; then
    HF_ARGS+=(--cache-dir "${HF_CACHE_DIR}")
  fi
  if [[ "${INCLUDE_DATASET}" == "1" ]]; then
    HF_ARGS+=(--include-dataset)
  fi
  if [[ "${INCLUDE_LLAMA3}" == "1" ]]; then
    HF_ARGS+=(--include-llama3)
  fi
  "${PYTHON_BIN}" scripts/download_hf_assets.py "${HF_ARGS[@]}"
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pytest tests -q
fi
