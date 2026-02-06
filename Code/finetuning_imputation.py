import os
import sys
import gc
import json
import random
from pathlib import Path
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- Imports from local modules ---
from model import TransformerModel_add_embedding
from loss import masked_mse_loss
from dataset import (
    prepare_imputation_data,
    ProximaImputationDataset,
    imputation_collate_fn,
    build_protein_knowledge_embedding
)
from utils import set_seed

# ===================================================================
# 1. Configuration (PROXIMA Imputation)
# ===================================================================
class ImputationConfig:
    task_name = "PROXIMA_Imputation"
    
    # Paths
    train_val_h5ad_path = "../Data/finetuning_data/Demo_data.h5ad"
    pretrained_model_path = "../Results/PROXIMA_Pretraining/20260108_142617/best_model.pt"
    vocab_path = "./Vocab/vocab.json"
    esm_embeddings_path = './Embedding/embeddings.npy'
    
    # Model Hyperparameters
    embsize = 768
    d_hid = 3072
    nlayers = 16
    nheads = 12
    n_layers_cls = 3
    dropout = 0.2
    
    # Training Hyperparameters
    epochs = 5
    batch_size = 128
    lr = 1e-5
    
    # Imputation Specifics
    imputation_masking_ratio = 0.2  
    loss_weight_cpr = 1.0           
    
    # Misc
    validation_split = 0.1
    num_workers = 4
    seed = 42

    # Tokens
    pad_token = "<pad>"
    cls_token = "<cls>"
    pad_value = 0

# ===================================================================
# 2. Main Execution
# ===================================================================

if __name__ == "__main__":
    config = ImputationConfig()
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PROXIMA Imputation Task: {config.task_name}")
    print(f"Device: {device}")

    # Results Directory
    results_base_dir = Path(f"../Results_finetune/{config.task_name}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = results_base_dir / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # --- Data Loading ---
    adata_tv, train_idx, val_idx, vocab = prepare_imputation_data(config)
    
    # --- Dataset & DataLoader ---
    train_dataset = ProximaImputationDataset(adata_tv, train_idx)
    val_dataset = ProximaImputationDataset(adata_tv, val_idx)

    collate_fn = partial(imputation_collate_fn, config=config, vocab=vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=config.num_workers)

    # --- Knowledge Embedding ---
    if os.path.exists(config.esm_embeddings_path):
        esm_embeddings_dict = np.load(config.esm_embeddings_path, allow_pickle=True).item()
        esm_embedding_dim = esm_embeddings_dict[next(iter(esm_embeddings_dict))].shape[0]
        esm_matrix_for_model = build_protein_knowledge_embedding(esm_embeddings_dict, vocab, esm_embedding_dim)
    else:
        print("Warning: ESM embeddings not found. Using random initialization.")
        esm_matrix_for_model = None

    # --- Model Initialization ---
    model = TransformerModel_add_embedding(
        ntoken=len(vocab), 
        d_model=config.embsize, 
        nhead=config.nheads,
        d_hid=config.d_hid, 
        nlayers=config.nlayers, 
        nlayers_cls=config.n_layers_cls,
        n_cls=1, # No classification head needed
        vocab=vocab, 
        dropout=config.dropout, 
        pad_token=config.pad_token,
        pad_value=config.pad_value, 
        do_mvc=True, # Enables CPR (Comprehensive Profile Reconstruction)
        use_fast_transformer=True,
        esm_embedding_matrix=esm_matrix_for_model
    )

    # --- Load Pre-trained Weights ---
    print(f"Loading PROXIMA pre-trained weights from {config.pretrained_model_path}")
    try:
        pretrained_state_dict = torch.load(config.pretrained_model_path, map_location='cpu')
        model.load_state_dict(pretrained_state_dict, strict=False)
        print("Successfully loaded pre-trained weights.")
    except Exception as e:
        print(f"Error loading weights: {e}. Training from scratch.")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # --- Training Loop ---
    best_val_loss = float('inf')

    print("\nStarting Training...")
    for epoch in range(config.epochs):
        # 1. Training Phase
        model.train()
        
        for batch in train_dataloader:
            genes = batch["genes"].to(device)
            expressions_truth = batch["expressions_truth"].to(device)
            expressions_masked_input = batch["expressions_masked_input"].to(device)
            imputation_eval_mask = batch["imputation_eval_mask"].to(device)
            
            src_key_padding_mask = torch.zeros(genes.shape, dtype=torch.bool).to(device)

            with autocast(enabled=torch.cuda.is_available()):
                output = model(
                    src=genes, 
                    values=expressions_masked_input, 
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=False, 
                    MVC=True 
                )
                
                # Calculate loss ONLY on the masked values (reconstruction)
                loss = masked_mse_loss(output["mvc_output"], expressions_truth, imputation_eval_mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        # 2. Validation Phase
        model.eval()
        total_val_loss = 0
        total_val_mae = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                genes = batch["genes"].to(device)
                expressions_truth = batch["expressions_truth"].to(device)
                expressions_masked_input = batch["expressions_masked_input"].to(device)
                imputation_eval_mask = batch["imputation_eval_mask"].to(device)
                
                src_key_padding_mask = torch.zeros(genes.shape, dtype=torch.bool).to(device)

                with autocast(enabled=torch.cuda.is_available()):
                    output = model(
                        src=genes, 
                        values=expressions_masked_input, 
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=False, 
                        MVC=True
                    )

                    loss = masked_mse_loss(output["mvc_output"], expressions_truth, imputation_eval_mask)
                    

                    if imputation_eval_mask.sum() > 0:
                        mae = torch.abs(output["mvc_output"][imputation_eval_mask] - expressions_truth[imputation_eval_mask]).mean()
                    else:
                        mae = torch.tensor(0.0, device=device)

                total_val_loss += loss.item()
                total_val_mae += mae.item()

        # Metrics & Logging
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_mae = total_val_mae / len(val_dataloader)
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{config.epochs} | Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_dir / "best_model_imputation.pt")
            print(f"  >> New best model saved (Val Loss: {best_val_loss:.4f})")

    print("\nPROXIMA Imputation Fine-tuning finished!")
    print(f"Best model saved to: {save_dir / 'best_model_imputation.pt'}")