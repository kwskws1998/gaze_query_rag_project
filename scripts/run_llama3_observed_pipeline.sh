#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ART="${ART:-artifacts/observed_regular_llama3}"
CONDITIONS="${CONDITIONS:-text mean_gaze actual_gaze shuffled_gaze}"
TOP_K="${TOP_K:-3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
GENERATOR_NAME="${GENERATOR_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
BASELINE_CONDITION="${BASELINE_CONDITION:-text}"
USE_HF_QA="${USE_HF_QA:-auto}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
RUN_ALIGNMENT="${RUN_ALIGNMENT:-1}"
RUN_EMBEDDINGS="${RUN_EMBEDDINGS:-1}"
RUN_RETRIEVAL="${RUN_RETRIEVAL:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EXCLUDE_REPEATED="${EXCLUDE_REPEATED:-1}"
INCLUDE_LLAMA3="${INCLUDE_LLAMA3:-1}"
RUN_TESTS="${RUN_TESTS:-0}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_PROJECT="${INSTALL_PROJECT:-1}"
DOWNLOAD_HF="${DOWNLOAD_HF:-1}"
INCLUDE_DATASET="${INCLUDE_DATASET:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-auto}"

find_ia_path() {
  local candidates=(
    "OneStop-Eye-Movements/data/OneStop/osfstorage-archive (1)/ia_Paragraph.csv.zip"
    "osfstorage-archive (1)/ia_Paragraph.csv.zip"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  local found
  found="$(find . -type f -name "ia_Paragraph.csv.zip" | sort | head -n 1)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found#./}"
    return 0
  fi
  return 1
}

QA_ARGS=()

resolve_qa_args() {
  local default_qa="OneStop-Eye-Movements/data_preprocessing/onestop_qa.json"
  case "${USE_HF_QA}" in
    1|true|yes)
      QA_ARGS=(--use-hf)
      ;;
    0|false|no)
      local qa_path="${QA_JSON_PATH:-${default_qa}}"
      if [[ ! -f "${qa_path}" ]]; then
        echo "QA JSON not found: ${qa_path}" >&2
        exit 1
      fi
      QA_ARGS=(--qa-json-path "${qa_path}")
      ;;
    auto)
      local qa_path="${QA_JSON_PATH:-${default_qa}}"
      if [[ -f "${qa_path}" ]]; then
        QA_ARGS=(--qa-json-path "${qa_path}")
      else
        QA_ARGS=(--use-hf)
      fi
      ;;
    *)
      echo "Unsupported USE_HF_QA value: ${USE_HF_QA}" >&2
      exit 1
      ;;
  esac
}

BOOTSTRAP_DOWNLOAD_DATA="${DOWNLOAD_DATA}"
if [[ "${BOOTSTRAP_DOWNLOAD_DATA}" == "auto" ]]; then
  if find_ia_path >/dev/null; then
    BOOTSTRAP_DOWNLOAD_DATA="0"
  else
    BOOTSTRAP_DOWNLOAD_DATA="1"
  fi
fi

if [[ "${RUN_BOOTSTRAP}" == "1" ]]; then
  INSTALL_TORCH="${INSTALL_TORCH}" \
  INSTALL_PROJECT="${INSTALL_PROJECT}" \
  DOWNLOAD_DATA="${BOOTSTRAP_DOWNLOAD_DATA}" \
  DOWNLOAD_HF="${DOWNLOAD_HF}" \
  INCLUDE_DATASET="${INCLUDE_DATASET}" \
  INCLUDE_LLAMA3="${INCLUDE_LLAMA3}" \
  RUN_TESTS="${RUN_TESTS}" \
  bash scripts/bootstrap_gpu_env.sh
fi

IA_PATH="${IA_PATH:-$(find_ia_path || true)}"
if [[ -z "${IA_PATH}" || ! -f "${IA_PATH}" ]]; then
  echo "Could not find ia_Paragraph.csv.zip. Set IA_PATH explicitly." >&2
  exit 1
fi

resolve_qa_args
read -r -a CONDITION_ARGS <<< "${CONDITIONS}"

ALIGN_ARGS=(
  --artifacts-dir "${ART}"
  --ia-path "${IA_PATH}"
)
if [[ "${EXCLUDE_REPEATED}" == "1" ]]; then
  ALIGN_ARGS+=(--exclude-repeated)
fi
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
  ALIGN_ARGS+=(--max-examples "${MAX_EXAMPLES}")
fi
if [[ -n "${MAX_READERS_PER_EXAMPLE:-}" ]]; then
  ALIGN_ARGS+=(--max-readers-per-example "${MAX_READERS_PER_EXAMPLE}")
fi

GEN_ARGS=(
  --retrieval-path "${ART}/retrieval/retrieval_results.jsonl"
  --artifacts-dir "${ART}"
  --generator-name "${GENERATOR_NAME}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --scoring-mode loglik
  --conditions "${CONDITION_ARGS[@]}"
)
if [[ -n "${MAX_RECORDS:-}" ]]; then
  GEN_ARGS+=(--max-records "${MAX_RECORDS}")
fi

echo "ART=${ART}"
echo "IA_PATH=${IA_PATH}"
echo "CONDITIONS=${CONDITIONS}"
echo "GENERATOR_NAME=${GENERATOR_NAME}"

if [[ "${RUN_ALIGNMENT}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/build_aligned_dataset.py "${ALIGN_ARGS[@]}" "${QA_ARGS[@]}"
fi

if [[ "${RUN_EMBEDDINGS}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/build_embeddings.py \
    --aligned-path "${ART}/data/aligned_examples.jsonl" \
    --artifacts-dir "${ART}" \
    --conditions "${CONDITION_ARGS[@]}" \
    --device "${DEVICE}"
fi

if [[ "${RUN_RETRIEVAL}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/run_retrieval.py \
    --artifacts-dir "${ART}" \
    --conditions "${CONDITION_ARGS[@]}" \
    --top-k "${TOP_K}"
fi

if [[ "${RUN_GENERATION}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/run_generation.py "${GEN_ARGS[@]}" "${QA_ARGS[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  "${PYTHON_BIN}" scripts/run_eval.py \
    --predictions-path "${ART}/predictions/predictions.jsonl" \
    --retrieval-path "${ART}/retrieval/retrieval_results.jsonl" \
    --artifacts-dir "${ART}" \
    --baseline-condition "${BASELINE_CONDITION}"
fi
