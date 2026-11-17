#!/usr/bin/env bash
# Step-0 eval of Moonlight-16B-A3B Base and Instruct on GSM8K test + MATH500.
# Needs 1 GPU. Runs four configs sequentially (model x prompt mode); a config whose
# tokenizer lacks a chat template exits early and the rest continue.
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python3 -c "import blobfile" 2>/dev/null || pip install --quiet --no-index \
    --find-links /inspire/hdd/project/qianghuaxuexi/public/wheels blobfile

MODEL_ROOT=${MODEL_ROOT:-/inspire/hdd3/project/qianghuaxuexi/public/models/moonshotai}
OUT_DIR=${OUT_DIR:-low_precision/outputs/moonlight_step0}
mkdir -p "${OUT_DIR}"

for tag in Moonlight-16B-A3B Moonlight-16B-A3B-Instruct; do
    for mode in chat raw; do
        echo "==== ${tag} / ${mode}"
        python low_precision/eval_moonlight_step0.py \
            --model "${MODEL_ROOT}/${tag}" \
            --tag "${tag}" \
            --prompt-mode "${mode}" \
            --out-dir "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/${tag}_${mode}.log"
    done
done

echo "==== summaries"
cat "${OUT_DIR}"/*_summary.json
