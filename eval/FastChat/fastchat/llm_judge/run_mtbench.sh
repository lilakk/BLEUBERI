#!/bin/bash

model_arg=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model)
            model_arg="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

models=(
    ""
)

export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [[ -n "$model_arg" ]]; then
    models=("$model_arg")
fi

for model in "${models[@]}"; do
    echo "====================================================="
    echo "Starting MT-Bench evaluation for model: $model"
    echo "====================================================="
    
    if [[ "$model" == *"ckpts"* ]]; then
        model_short=$(echo $model | rev | cut -d/ -f1,2 | rev | tr '/' '_')
    else
        model_short=$(basename $model)
    fi

    python gen_model_answer_vllm.py --model-path $model --model-id $model_short
    python gen_judgment.py --model-list $model_short --parallel 10

    echo "MT-Bench evaluation completed for: $model"
    echo "====================================================="
    echo ""
done

python show_result.py