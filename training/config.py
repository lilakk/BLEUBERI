import argparse
from typing import Optional, Union, List

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration arguments")
    
    # DeepSpeed configuration
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Used by Deepspeed for distributed training")
    
    # Output and run configuration
    parser.add_argument("--run_name", type=str, default="default", help="Name of the training run")
    parser.add_argument("--wandb_project", type=str, default="default", help="Name of the wandb project")
    
    # Model and cache configuration
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--cache_dir", type=str, required=True, help="Cache directory for models and data")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory for model checkpoints")
    
    # Data configuration
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument("--load_from_disk", action="store_true", help="Whether to load data from disk")
    parser.add_argument("--shuffle", action="store_true", help="Whether to shuffle the data")
    parser.add_argument("--streaming", action="store_true", help="Whether to stream the data")
    parser.add_argument("--train_split", type=str, default="train", help="Training split name")
    parser.add_argument("--model_max_length", type=int, default=128000, help="Maximum model sequence length")
    
    # Training configuration
    parser.add_argument("--save_strategy", type=str, required=True, help="Save strategy (steps or epoch)")
    parser.add_argument("--save_steps", type=int, help="Save frequency in steps")
    parser.add_argument("--save_total_limit", type=int, default=10, help="Maximum number of checkpoints to keep")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy (steps or epoch)")
    parser.add_argument("--eval_steps", type=float, default=0.1, help="Evaluation frequency (as ratio or absolute steps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", help="Whether to use bf16 precision")
    
    # Learning rate and optimization
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup", help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=0.2, help="Maximum gradient norm")
    
    # Batch configuration
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Per device training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Per device evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    
    # Generation configuration
    parser.add_argument("--reasoning", action="store_true", help="Whether to add reasoning to the data")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=512, help="Maximum completion length")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for generation")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of iterations")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="info", help="Logging level")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging frequency in steps")

    # SFT configuration
    parser.add_argument("--packing", action="store_true", help="Whether to pack the data")
    parser.add_argument("--dataset_text_field", type=str, default="text", help="Dataset text field")
    
    # Reward functions
    parser.add_argument("--reward_funcs", nargs="+", choices=["bleu", "emb", "bertscore", "rm", "format", "bleu_emb", "rouge", "bleu_rouge_f1", "bleu_rm"], 
                        help="List of reward functions to use (e.g., --reward_funcs bleu emb)")

    return parser.parse_args()
