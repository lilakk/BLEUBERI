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
    "meta-llama/Llama-3.1-8B-Instruct"
)

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export GPT_EVAL_NAME="gpt-4.1-mini"  # TODO: change accordingly!

if [[ -n "$model_arg" ]]; then
    models=("$model_arg")
fi

for model in "${models[@]}"; do
    echo "====================================================="
    echo "Starting WildBench evaluation for model: $model"
    echo "====================================================="
    
    if [[ "$model" == *"ckpts"* ]]; then
        model_short=$(echo $model | rev | cut -d/ -f1,2 | rev | tr '/' '_')
    else
        model_short=$(basename $model)
    fi

    bash scripts/_common_vllm.sh $model $model_short 1
    bash evaluation/run_eval_v2_instant.score.sh $model_short

    echo "WildBench evaluation completed for: $model"
    echo "====================================================="
    echo ""
done

bash leaderboard/show_eval.sh score_only