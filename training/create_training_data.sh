#!/bin/bash

# to only use the Tulu3 reference (default setup in our main experiments):
python create_training_data.py grpo \
    --hf_dataset_path yapeichang/BLEUBERI-Tulu3-50k \
    --ref_models gold \
    --selection_mode hard \
    --model Qwen/Qwen2.5-7B \
    --metric bleu \
    --num_examples 5000

# to use 5 references:
python create_training_data.py grpo \
    --hf_dataset_path yapeichang/BLEUBERI-Tulu3-50k \
    --ref_models gold claude-3-7-sonnet@20250219 deepseek-chat-v3 gemini-2.5-pro-exp-03-25 o4-mini-2025-04-16 \
    --selection_mode hard \
    --model Qwen/Qwen2.5-7B \
    --metric bleu \
    --num_examples 5000

# to score the data using RM-8B instead of BLEU:
python create_training_data.py grpo \
    --hf_dataset_path yapeichang/BLEUBERI-Tulu3-50k \
    --selection_mode hard \
    --model Qwen/Qwen2.5-7B \
    --metric rm \
    --num_examples 5000

# to create SFT data based on an existing GRPO training dataset:
python create_training_data.py sft \
    --input_data_path ../data/data_grpo/BLEUBERI-Tulu3-50k_bleu_Qwen2.5-7B_5ref-gold-claude-deepseek-gemini-o4mini_hard_5000

python create_training_data.py sft \
    --input_data_path ../data/data_grpo/BLEUBERI-Tulu3-50k_bleu_Qwen2.5-7B_1ref-gold_hard_5000
