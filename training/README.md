# GRPO and SFT training

## Creating training data

To create training data for a specific run based on the [50K data pool](https://huggingface.co/datasets/yapeichang/BLEUBERI-Tulu3-50k), we use [`create_training_data.py`](create_training_data.py).

This script supports two main commands: `grpo` and `sft`.

### `grpo` command
The `grpo` command prepares data for GRPO (Group Relative Policy Optimization) training. It can:
- Aggregate reference outputs from different models specified by `--ref_models`.
  - Available options include: `gold` (default), `claude-3-7-sonnet@20250219`, `deepseek-chat-v3`, `gemini-2.5-pro-exp-03-25`, `o4-mini-2025-04-16`, `Llama-3.1-8B-Instruct`.
- Score model outputs using a specified `--metric` (e.g., `bleu`, `rm`, `rouge`) and a `--model` for generation if outputs are not cached.
- Select a subset of examples using `--selection_mode` (`random`, `easy`, `medium`, `hard`) and `--num_examples`.
  - The default setting is `hard`, which will select the `num_examples` lowest scoring examples from the 50K data pool.
- Specify the source HuggingFace dataset with `--hf_dataset_path`. The default is `yapeichang/BLEUBERI-Tulu3-50k`.
- Control output directory with `--output_dir` and the output dataset name with `--output_dataset_name`.

Example usage:
```bash
python training/create_training_data.py grpo \
    --hf_dataset_path yapeichang/BLEUBERI-Tulu3-50k \
    --ref_models gold \
    --selection_mode hard \
    --num_examples 1000 \
    --metric bleu \
    --model Qwen/Qwen2.5-7B \
    --output_dir ../data \
    --output_dataset_name my_grpo_dataset
```

### `sft` command
The `sft` command prepares data for SFT training. It converts a dataset (often produced by the `grpo` command) into a series of prompt-response pairs.
- Specify the input data path (a directory containing a HuggingFace dataset) with `--input_data_path`.
- Control output directory with `--output_dir`.
    
Example usage:
```bash
python training/create_training_data.py sft \
    --input_data_path ../data/data_grpo/my_grpo_dataset \
    --output_dir ../data
```

The script also handles on-the-fly inference if model outputs are missing for scoring in the `grpo` command, using vLLM. Arguments like `--inference_max_new_tokens` and `--seed` can be used to control this process.

## GRPO training

To run GRPO training, see [`grpo.sh`](grpo.sh) for an example job script that covers training data creation, specifying DeepSpeed config, and launching the training job.

## SFT training

To run SFT training, see [`sft.sh`](sft.sh) for an example job script that covers training data creation, specifying DeepSpeed config, and launching the training job.