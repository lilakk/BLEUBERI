#!/bin/bash

# -------------------------------
# Working Directory and Environment  # TODO: change to your own!
# -------------------------------

# conda activate bleuberi

WORK_DIR="/mnt/sharedfs/yapei/BLEUBERI/training"
cd $WORK_DIR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/sharedfs/yapei/venvs/rlit

# --------------------------------
# Creating training data  # TODO: change to your own!
# --------------------------------

grpo_dataset_name="BLEUBERI-Tulu3-50k_bleu_Qwen2.5-7B_1ref-gold_hard_5000"

# the data will be saved to ../data/data_sft/${grpo_dataset_name}_SFT
python create_training_data.py sft \
    --input_data_path ../data/data_grpo/$grpo_dataset_name

# -------------------------------
# Training parameters  # TODO: change to your own!
# -------------------------------

out_dir="../ckpts"
cache_dir="/mnt/sharedfs/cache/hub"
run_prefix=""

wandb_project="BLEUBERI"
run_name="qwen7b_SFT_1ref-gold_5k"

model="Qwen/Qwen2.5-7B"
data_path="../data/data_sft/${grpo_dataset_name}_SFT"

per_device_train_batch_size=4
gradient_accumulation_steps=8

lr=5e-6
lr_scheduler_type="constant_with_warmup"

num_epochs=1
max_steps=1  # if not -1, will override num_epochs
save_strategy="steps"
save_steps=250

warmup_ratio=0.05
max_grad_norm=0.2

train_split="train"
model_max_length=128000

packing=false
dataset_text_field="text"

save_total_limit=20
seed=42
bf16=true

NUM_GPUS=$(nvidia-smi -L | wc -l)
global_bsz=$((per_device_train_batch_size * gradient_accumulation_steps * NUM_GPUS))
data_base_name=$(basename $data_path)
model_base_name=$(basename $model)

if [ -z "$run_name" ]; then
    run_name="${run_prefix}_${model_base_name}_${data_base_name}_lr${lr}_bsz${global_bsz}_epochs${num_epochs}"
fi

ckpt_dir=${out_dir}/${run_name}
mkdir -p $ckpt_dir

cp "$0" "$ckpt_dir/$(basename $0)"
rsync -av --exclude="wandb" --exclude="wandb_tables" --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" --exclude="*.csv" $WORK_DIR/ $ckpt_dir/training/

# -------------------------------
# DeepSpeed configuration
# -------------------------------

cat << EOF > ${ckpt_dir}/ds_config.json
{
    "bf16": {
        "enabled": "auto"
    },
    "activation_checkpointing": {
        "partition_activations": true,
        "cpu_checkpointing": true,
        "contiguous_memory_optimization": true,
        "checkpoint_in_cpu": true,
        "profile": false
    },
    "zero_optimization": {
        "stage": 3,
        "stage3_gather_16bit_weights_on_model_save": true,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": ${gradient_accumulation_steps},
    "gradient_clipping": ${max_grad_norm},
    "train_batch_size": ${global_bsz},
    "train_micro_batch_size_per_gpu": ${per_device_train_batch_size},
    "wall_clock_breakdown": false
}
EOF

# -------------------------------
# Logging and setup
# -------------------------------

echo "=== System Information ===" | tee -a $ckpt_dir/setup.log
echo "Starting job at $(date)" | tee -a $ckpt_dir/setup.log
echo "Working directory: $WORK_DIR" | tee -a $ckpt_dir/setup.log
echo "Host: $(hostname)" | tee -a $ckpt_dir/setup.log
nvidia-smi | tee -a $ckpt_dir/setup.log

arguments=(
    --run_name $run_name
    --model_path $model
    --cache_dir $cache_dir
    --ckpt_dir $ckpt_dir
    --data_path $data_path
    --train_split $train_split
    --model_max_length $model_max_length
    --num_epochs $num_epochs
    --max_steps $max_steps
    --save_strategy $save_strategy
    --save_steps $save_steps
    --save_total_limit $save_total_limit
    --seed $seed
    --learning_rate $lr
    --lr_scheduler_type $lr_scheduler_type
    --warmup_ratio $warmup_ratio
    --max_grad_norm $max_grad_norm
    --per_device_train_batch_size $per_device_train_batch_size
    --gradient_accumulation_steps $gradient_accumulation_steps
    --log_level info
    --logging_steps 1
    --deepspeed ${ckpt_dir}/ds_config.json
)

# Only add boolean flags if they are True
[[ "$packing" == true ]] && arguments+=(--packing)
[[ "$bf16" == true ]] && arguments+=(--bf16)

echo "Arguments: ${arguments[*]}" | tee -a $ckpt_dir/setup.log

# -------------------------------
# Launch DeepSpeed on current node
# -------------------------------

echo "Launching DeepSpeed on $(hostname)" | tee -a $ckpt_dir/setup.log

deepspeed_cmd="deepspeed \
    --num_gpus ${NUM_GPUS} \
    ${WORK_DIR}/sft.py ${arguments[*]}"

echo "Command: $deepspeed_cmd" | tee -a $ckpt_dir/setup.log

$deepspeed_cmd 2>&1 | tee -a $ckpt_dir/node_$(hostname).log

echo "===========================================" | tee -a $ckpt_dir/setup.log
echo "Training process completed!" | tee -a $ckpt_dir/setup.log
echo "===========================================" | tee -a $ckpt_dir/setup.log
echo "Checkpoint directory: $ckpt_dir"
echo "Log file: $ckpt_dir/node_$(hostname).log"
