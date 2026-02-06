import os
import sys
import gc
import json
import random
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# --- Path Setup ---
# Add parent directory to sys.path to allow importing src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- Imports from local modules ---
from model import TransformerModel_add_embedding  
from loss import masked_mse_loss
from dataset import (
    prepare_annotation_data,
    PhenotypeAnnotationDataset,
    annotation_collate_fn,
    annotation_eval_collate_fn,
    build_protein_knowledge_embedding
)
from utils import set_seed, calculate_metrics

# ===================================================================
# 1. Configuration (PROXIMA Fine-tuning)
# ===================================================================
class FinetuneConfig:
    task_name = "PROXIMA_Annotation"
    
    # Paths (Update these as needed)
    train_val_h5ad_path = "../Data/finetuning_data/Demo_train.h5ad"
    test_h5ad_path = "../Data/finetuning_data/Demo_test.h5ad"
    pretrained_model_path = "../Results/PROXIMA_Pretraining/20260108_142617/best_model.pt"
    vocab_path = "./Vocab/vocab.json"
    esm_embeddings_path = './Embedding/embeddings.npy'
    
    cell_type_key = "cell_type"
    seed = 42  # Fixed seed for single run

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
    lr = 5e-6
    max_length = 400
    
    # Loss Weights
    loss_weight_cls = 1.0
    loss_weight_mep = 0.2
    loss_weight_cpr = 0.2

    # Misc
    validation_split = 0.1
    num_workers = 4
    
    # Tokens
    special_tokens = ["<pad>", "<cls>", "<eoc>", "<mask>"]
    pad_token = "<pad>"
    mask_token = "<mask>"
    cls_token = "<cls>"
    pad_value = 0
    mask_value = -1

# ===================================================================
# 2. Main Execution
# ===================================================================

if __name__ == "__main__":
    config = FinetuneConfig()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PROXIMA Annotation Task: {config.task_name}")
    print(f"Device: {device}")

    # Results Directory
    results_base_dir = Path(f"../Results_finetune/{config.task_name}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = results_base_dir / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save_dir = str(save_dir)

    # --- Data Loading ---
    # Using helper from dataset.py
    adata_tv, adata_test, train_idx, val_idx, vocab, cell_type_map, inverse_cell_type_map = prepare_annotation_data(config)
    config.mask_id = vocab[config.mask_token]

    # --- Dataset & DataLoader ---
    train_dataset = PhenotypeAnnotationDataset(adata_tv, train_idx, vocab, config)
    val_dataset = PhenotypeAnnotationDataset(adata_tv, val_idx, vocab, config)
    test_dataset = PhenotypeAnnotationDataset(adata_test, np.arange(adata_test.n_obs), vocab, config)

    collate_fn = partial(annotation_collate_fn, config=config, vocab=vocab)
    collate_fn_eval = partial(annotation_eval_collate_fn, config=config, vocab=vocab)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_eval, num_workers=config.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn_eval, num_workers=config.num_workers)

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
        n_cls=config.num_cell_types, 
        vocab=vocab, 
        dropout=config.dropout, 
        pad_token=config.pad_token,
        pad_value=config.pad_value, 
        do_mvc=True, # Enables CPR
        use_batch_labels=False,
        do_dab=False,
        use_fast_transformer=True,
        esm_embedding_matrix=esm_matrix_for_model
    )

    # --- Load Pre-trained Weights ---
    print(f"Loading PROXIMA pre-trained weights from {config.pretrained_model_path}")
    try:
        pretrained_state_dict = torch.load(config.pretrained_model_path, map_location='cpu')
        model_dict = model.state_dict()
        # Filter matching keys
        pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    except Exception as e:
        print(f"Error loading weights: {e}. Training from scratch.")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # --- Training Loop ---
    best_val_metric = -1.0
    patience = 10
    patience_counter = 0

    print("\nStarting Training...")
    for epoch in range(config.epochs):
        # 1. Training Phase
        model.train()
        for batch in train_dataloader:
            genes = batch["genes"].to(device)
            expressions_truth = batch["expressions_truth"].to(device)
            expressions_masked = batch["expressions_masked"].to(device)
            mlm_mask = batch["mlm_mask"].to(device)
            cell_type_ids = batch["cell_type_ids"].to(device)
            src_key_padding_mask = (genes == vocab[config.pad_token])

            with autocast(enabled=torch.cuda.is_available()):
                output = model(
                    src=genes, 
                    values=expressions_masked, 
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=True, 
                    MVC=True
                )
                loss_cls = criterion(output["cls_output"], cell_type_ids)
                loss_mep = masked_mse_loss(output["mlm_output"], expressions_truth, mlm_mask)
                loss_cpr = masked_mse_loss(output["mvc_output"], expressions_truth, ~src_key_padding_mask)
                
                combined_loss = (config.loss_weight_cls * loss_cls + 
                                 config.loss_weight_mep * loss_mep + 
                                 config.loss_weight_cpr * loss_cpr)

            optimizer.zero_grad()
            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
        # 2. Validation Phase
        model.eval()
        all_val_preds, all_val_labels = [], []
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                genes = batch["genes"].to(device)
                expressions_truth = batch["expressions_truth"].to(device)
                expressions_masked = batch["expressions_masked"].to(device)
                mlm_mask = batch["mlm_mask"].to(device)
                cell_type_ids = batch["cell_type_ids"].to(device)
                src_key_padding_mask = (genes == vocab[config.pad_token])

                with autocast(enabled=torch.cuda.is_available()):
                    output = model(src=genes, values=expressions_masked, src_key_padding_mask=src_key_padding_mask, CLS=True, MVC=True)
                    loss_cls = F.cross_entropy(output["cls_output"], cell_type_ids)
                    loss_mep = masked_mse_loss(output["mlm_output"], expressions_truth, mlm_mask)
                    loss_cpr = masked_mse_loss(output["mvc_output"], expressions_truth, ~src_key_padding_mask)
                    combined_loss = (config.loss_weight_cls * loss_cls + config.loss_weight_mep * loss_mep + config.loss_weight_cpr * loss_cpr)
                
                total_val_loss += combined_loss.item()
                all_val_preds.append(torch.argmax(output["cls_output"], dim=1).cpu())
                all_val_labels.append(cell_type_ids.cpu())

        # Metrics
        val_metrics = calculate_metrics(torch.cat(all_val_labels).numpy(), torch.cat(all_val_preds).numpy())
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        scheduler.step(val_metrics["macro_f1"])

        print(f"Epoch {epoch+1}/{config.epochs} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_metrics['macro_f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        # Checkpoint
        if val_metrics['macro_f1'] > best_val_metric:
            best_val_metric = val_metrics['macro_f1']
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model_finetuned.pt")
            print("  >> New best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  >> Early stopping triggered.")
                break

    # --- Final Testing ---
    print("\nTesting")
    best_model_path = save_dir / "best_model_finetuned.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        all_test_preds, all_test_labels = [], []

        with torch.no_grad():
            for batch in test_dataloader:
                genes = batch["genes"].to(device)
                expressions_masked = batch["expressions_masked"].to(device)
                cell_type_ids = batch["cell_type_ids"].to(device)
                src_key_padding_mask = (genes == vocab[config.pad_token])

                with autocast(enabled=torch.cuda.is_available()):
                    output = model(src=genes, values=expressions_masked, src_key_padding_mask=src_key_padding_mask, CLS=True, MVC=True)
                
                all_test_preds.append(torch.argmax(output["cls_output"], dim=1).cpu())
                all_test_labels.append(cell_type_ids.cpu())

        all_test_preds = torch.cat(all_test_preds).numpy()
        all_test_labels = torch.cat(all_test_labels).numpy()
        test_metrics = calculate_metrics(all_test_labels, all_test_preds)

        print(f"Test Results: Accuracy: {test_metrics['accuracy']:.4f}, Macro F1: {test_metrics['macro_f1']:.4f}")
        
        # Save Predictions
        results_df = pd.DataFrame({
            'cell_id': adata_test.obs.index,
            'true_cell_type': [inverse_cell_type_map[i] for i in all_test_labels],
            'predicted_cell_type': [inverse_cell_type_map[i] for i in all_test_preds]
        })
        results_df.to_csv(save_dir / "test_predictions.csv", index=False)
        
        # Save Metrics
        metrics_df = pd.DataFrame([test_metrics])
        metrics_df.to_csv(save_dir / "test_metrics.csv", index=False)
        print(f"Predictions and metrics saved to {save_dir}")

    # Cleanup
    del model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()
    gc.collect()