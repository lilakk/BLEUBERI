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

if [[ -n "$model_arg" ]]; then
    models=("$model_arg")
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn

for model in "${models[@]}"; do
    echo "====================================================="
    echo "Starting Arena-Hard evaluation for model: $model"
    echo "====================================================="
    
    python3 gen_answer_local.py --model "$model"

    if [[ "$model" == *"ckpts"* ]]; then
        model_short=$(echo $model | rev | cut -d/ -f1,2 | rev | tr '/' '_')
    else
        model_short=$(basename $model)
    fi
    python3 gen_judgment.py --model "$model_short"
    python3 add_markdown_info.py --dir data/arena-hard-v0.1/model_answer --output-dir data/arena-hard-v0.1/model_answer
    
    echo "Arena-Hard evaluation completed for: $model"
    echo "====================================================="
    echo ""
done

python3 show_result.py --style-control --load-bootstrap --load-battles