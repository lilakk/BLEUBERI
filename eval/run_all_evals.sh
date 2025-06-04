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

if [[ -n "$model_arg" ]]; then
    models=("$model_arg")
fi

for model in "${models[@]}"; do
    cd FastChat/fastchat/llm_judge
    bash run_mtbench.sh --model $model
    cd ../../..

    cd WildBench
    bash run_wildbench_instant.sh --model $model
    cd ..

    cd arena-hard
    bash run_arenahard.sh --model $model
    cd ..

    cd arena-hard-v2.0
    bash run_arenahard_2.0.sh --model $model
    cd ..
done