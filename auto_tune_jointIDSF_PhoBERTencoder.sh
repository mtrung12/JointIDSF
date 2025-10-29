#!/bin/bash
# auto_tune_jointIDSF.sh

# ---- Stage 1: train base encoder once ----
bash run_jointBERT-CRF_PhoBERTencoder.sh

# Best pretrained checkpoint path (adjust if you test multiple Stage1 runs)
PRETRAINED_PATH="JointBERT-CRF_PhoBERTencoder/3e-5/0.6/100"

# ---- Stage 2: automatic tuning ----
lrs=(2e-5 3e-5 4e-5)
intent_loss=(0.15 0.3 0.6)
bss=(16 32)

BASE_DIR="JointIDSF_tuning"
mkdir -p "$BASE_DIR"

for lr in "${lrs[@]}"; do
  for c in "${intent_loss[@]}"; do
    for bs in "${bss[@]}"; do
      MODEL_DIR="${BASE_DIR}/lr${lr}_c${c}_bs${bs}"
      echo ">>> Running JointIDSF: LR=${lr}, c=${c}, BS=${bs}"
      mkdir -p "${MODEL_DIR}"

      python3 main.py \
        --token_level word-level \
        --model_type phobert \
        --model_dir "${MODEL_DIR}" \
        --data_dir PhoATIS \
        --seed 100 \
        --do_train \
        --do_eval_dev \
        --save_preds \
        --preds_output_dir ./preds \
        --save_steps 140 \
        --logging_steps 140 \
        --num_train_epochs 50 \
        --tuning_metric mean_intent_slot \
        --use_intent_context_attention \
        --attention_embedding_size 200 \
        --use_crf \
        --gpu_id 0 \
        --embedding_type soft \
        --intent_loss_coef "${c}" \
        --pretrained \
        --pretrained_path "${PRETRAINED_PATH}" \
        --learning_rate "${lr}" \
        --train_batch_size "${bs}" \
        | tee "${MODEL_DIR}/log.txt"

      # quick metric extraction
      grep "mean_intent_slot" "${MODEL_DIR}/log.txt" >> "${BASE_DIR}/summary.txt"
    done
  done
done

echo "All tuning runs completed. Summary saved to ${BASE_DIR}/summary.txt"
