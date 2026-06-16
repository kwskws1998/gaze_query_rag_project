# GPU Setup

This repo is set up for a one-shot GPU server bootstrap after `git clone`.

## 1. Clone

```bash
git clone <YOUR_REPO_URL> gaze_query_rag_project
cd gaze_query_rag_project
```

## 2. Create Environment

Conda:

```bash
conda create -n gaze-rag python=3.11 -y
conda activate gaze-rag
```

Plain venv:

```bash
CREATE_VENV=1 bash scripts/bootstrap_gpu_env.sh
source .venv/bin/activate
```

## 3. One-Shot Install, Data Download, Model Cache

Default project data bundle:

```bash
HF_TOKEN="<YOUR_HF_TOKEN>" \
bash scripts/bootstrap_gpu_env.sh
```

The default Google Drive file is:

```text
https://drive.google.com/file/d/1lvPnyMV_AbKo2CdcAy3kvQa8qIJuVSeg/view?usp=sharing
```

The bootstrap script downloads it through:

```text
https://drive.google.com/uc?id=1lvPnyMV_AbKo2CdcAy3kvQa8qIJuVSeg
```

Override with another Google Drive file URL:

```bash
HF_TOKEN="<YOUR_HF_TOKEN>" \
GDRIVE_URL="<YOUR_GOOGLE_DRIVE_SHARE_URL>" \
bash scripts/bootstrap_gpu_env.sh
```

Override with another Google Drive file ID:

```bash
HF_TOKEN="<YOUR_HF_TOKEN>" \
GDRIVE_ID="<YOUR_GOOGLE_DRIVE_FILE_ID>" \
bash scripts/bootstrap_gpu_env.sh
```

By default this installs CUDA 12.1 PyTorch, installs the package editable, downloads `intfloat/e5-large-v2`, downloads the smoke-test generator `HuggingFaceTB/SmolLM2-135M-Instruct`, caches `malmaud/onestop_qa`, and runs tests.

## 4. Optional Llama 3 Cache

`meta-llama/Meta-Llama-3-8B-Instruct` is gated. After accepting access on Hugging Face, run:

```bash
HF_TOKEN="<YOUR_HF_TOKEN>" \
INCLUDE_LLAMA3=1 \
RUN_TESTS=0 \
bash scripts/bootstrap_gpu_env.sh
```

The main config target remains Llama 3. The small generator is only for smoke tests.

## 5. Useful Switches

```bash
RUN_TESTS=0 bash scripts/bootstrap_gpu_env.sh
DOWNLOAD_DATA=0 bash scripts/bootstrap_gpu_env.sh
DOWNLOAD_HF=0 bash scripts/bootstrap_gpu_env.sh
INSTALL_TORCH=0 bash scripts/bootstrap_gpu_env.sh
HF_CACHE_DIR="/workspace/.cache/huggingface" bash scripts/bootstrap_gpu_env.sh
```

## 6. Expected Data Layout

After downloading or extracting the data bundle, these paths should exist:

```text
OneStop-Eye-Movements/data/OneStop/osfstorage-archive (1)/ia_Paragraph.csv.zip
OneStop-Eye-Movements/data/OneStop/osfstorage-archive (1)/fixations_Paragraph.csv.zip
OneStop-Eye-Movements/data_preprocessing/onestop_qa.json
```

## 7. First Checks

```bash
python -m pytest tests -q
python scripts/download_hf_assets.py --skip-generator --include-dataset
```

For GPU:

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 8. Llama 3 Generation On RTX 4090

```bash
python scripts/run_generation.py \
  --retrieval-path artifacts/text_baseline_full/retrieval/retrieval_results.jsonl \
  --artifacts-dir artifacts/text_baseline_full_llama3 \
  --qa-json-path OneStop-Eye-Movements/data_preprocessing/onestop_qa.json \
  --generator-name meta-llama/Meta-Llama-3-8B-Instruct \
  --device cuda \
  --dtype float16 \
  --scoring-mode loglik
```
