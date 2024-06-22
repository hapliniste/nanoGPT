# train_c.py
# This script extends train.py to include doubling the model's width after max_iters

import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT, GPTConfig, duplicate_and_scale_weights

print("### TRAIN CUSTOM - IMPORTS")

# Load configurations and utilities from train.py
from train import setup_training, get_batch, run_training_loop, parse_arguments

def modify_model_for_double_width(model):
    """Double the width of the transformer model by duplicating and scaling weights."""
    for block in model.transformer.h:
        # Double and scale weights for each component in the block
        block.attn.c_attn = duplicate_and_scale_weights(block.attn.c_attn)
        block.attn.c_proj = duplicate_and_scale_weights(block.attn.c_proj)
        block.mlp.c_fc = duplicate_and_scale_weights(block.mlp.c_fc)
        block.mlp.c_proj = duplicate_and_scale_weights(block.mlp.c_proj)
    
    # Adjust the final layernorm and linear layers
    config = model.config
    new_embd_size = config.n_embd * 2
    model.transformer.ln_f = LayerNorm(new_embd_size, bias=config.bias)
    model.lm_head = nn.Linear(new_embd_size, config.vocab_size, bias=False)
    model.transformer.wte.weight = model.lm_head.weight  # Tie weights

    print("Model width doubled successfully.")
    return model

def main():
    print("### TRAIN CUSTOM - MAIN")
    args = parse_arguments()
    
    # Setup training configuration, model, optimizer, etc.
    device, model, optimizer, train_data_loader, config = setup_training(args)

    # Optionally load from checkpoint
    if args.resume_checkpoint:
        model, optimizer = load_model_from_checkpoint(model, optimizer, args.resume_checkpoint)
        print("### MODEL LOADED")

    # Detect if max_iters reached and modify the model
    if args.iter_num >= args.max_iters:
        print("### MODIFYING MODEL")
        model = modify_model_for_double_width(model)
        print("### MODEL MODIFYED")
        if args.ddp:  # If using Distributed Data Parallel, need to re-wrap the model
            model = DDP(model, device_ids=[args.local_rank])

    # Run training loop
    print("### RUN TRAINING")
    run_training_loop(model, optimizer, train_data_loader, config)

if __name__ == "__main__":
    main()
