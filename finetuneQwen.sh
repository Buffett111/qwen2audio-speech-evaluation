#!/bin/bash
#
# 用途：自動化微調 Qwen2‑Audio 7B Instruct 模型在口說評量資料集上的程序。
#
# 使用者應先安裝並配置好 Python 環境（例如 conda 環境），並確保安裝了 transformers、peft、datasets、torch、scipy 等套件。
# 此腳本假設已經使用 `huggingface‑cli login` 登入，以便載入私有資料集【355行】。
#
# 使用範例：
#   bash finetuneQwen.sh \
#       --dataset ntnu-smil/sla-p1 \
#       --loss_type FA \
#       --output ./finetuned_model
#
# 如果您要改變其他超參數（如學習率、batch size），可於此處修改。

set -e

DATASET="ntnu-smil/sla-p1"
LOSS_TYPE="FA"
OUTPUT_DIR="./finetuned_qwen2audio"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --loss_type)
      LOSS_TYPE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# 建議使用 GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python train_qwen2_audio.py \
  --dataset_name "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --loss_type "$LOSS_TYPE" \
  --per_device_batch_size 8 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --val_split

echo "Finetuning completed. Model saved in $OUTPUT_DIR"