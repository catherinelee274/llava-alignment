#!/bin/bash

python trainPA.py \
    --alignment_method orpo \
    --save_model "Model/blip-vanilla-mllm_orpo" \
    --epochs 10 \
    --batchsize 16 \
    --lr 8e-6 \
    --train_percentage 90 \
    --eval_percentage 10 \
    --max_eval_llm 100 \
    --iterations_loss 25 \
    --dataset MLLM-DPO \
    --mllm-dpo-json ./data/dpo_inference0104.with_logpllava-v1.5-13b_2024-02-03.json \
    --mllm-dpo-images ./data/ \
    --wandb True \
    --wandb_project cs231n_blip-vqa_report \
2>&1 | tee orpo-report.log
