import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
import random
from scipy.spatial import cKDTree
from functools import partial
import time
from datetime import datetime
import math
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, broadcast_object_list

import sys
sys.path.append("../")

from model import TransformerModel_add_embedding
from loss import masked_mse_loss, spatially_aware_contrastive_loss
from dataset import (
    MicroenvironmentAwarePatchDataset, 
    microenvironment_collate_fn, 
    build_protein_knowledge_embedding
)
from Src.tokenizer import GeneVocab


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision="fp16",
    kwargs_handlers=[ddp_kwargs]
)

# ===================================================================
# 1. Configuration (PROXIMA Hyperparameters)
# ===================================================================
class ProximaConfig:
    task_name = "PROXIMA_Pretraining"
    data_path = "../data"
    vocab_path = "./Vocab/vocab.json"
    esm_embeddings_path = './Embedding/embeddings.npy'
    
    validation_split = 0.1
    patch_size = 32
    
    special_tokens = ["<pad>", "<cls>", "<eoc>", "<mask>"]
    pad_token = "<pad>"
    mask_token = "<mask>"
    cls_token = "<cls>"
    pad_value = 0
    mask_value = -1

    embsize = 768
    d_hid = 3072
    nlayers = 16
    nheads = 12
    n_layers_cls = 3
    dropout = 0.2
    
    impute_mvc_knn_k = 10
    
    epochs = 5
    batch_size = 16
    lr = 5e-5
    max_length = 400

    warmup_ratio = 0.05
    lr_end_factor = 0.1

    loss_weight_mep = 1.0      
    loss_weight_cpr = 1.0      
    loss_weight_nei = 1.0       
    loss_weight_scl = 0.1       
    
    scl_temperature = 0.2
    scl_k_negatives = 64

    num_workers = 4

config = ProximaConfig()

results_base_dir = Path(f"../results/{config.task_name}")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = results_base_dir / timestamp

if accelerator.is_main_process:
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"PROXIMA: Results will be saved to: {save_dir}")

config.save_dir = str(save_dir) 

# ===================================================================
# 2. Vocabulary and Knowledge Embedding
# ===================================================================


if os.path.exists(config.vocab_path):
    vocab = GeneVocab.from_file(config.vocab_path)
else:
    raise FileNotFoundError(f"Vocabulary not found at {config.vocab_path}")

config.mask_id = vocab[config.mask_token]

esm_matrix_for_model = None
if accelerator.is_main_process:
    if os.path.exists(config.esm_embeddings_path):
        esm_embeddings_dict = np.load(config.esm_embeddings_path, allow_pickle=True).item()
        esm_embedding_dim = esm_embeddings_dict[next(iter(esm_embeddings_dict))].shape[0]
        
        esm_matrix_for_model = build_protein_knowledge_embedding(
            esm_embeddings_dict, 
            vocab, 
            esm_embedding_dim,
        )
    else:
        print("Warning: ESM embeddings file not found. Initializing randomly.")

object_list_to_broadcast = [esm_matrix_for_model]
accelerator.wait_for_everyone()
broadcast_object_list(object_list_to_broadcast, from_process=0)
esm_matrix_for_model = object_list_to_broadcast[0]

if accelerator.is_main_process:
    vocab_save_path = Path(config.save_dir) / "vocab.json"
    vocab.save_json(vocab_save_path)

# ===================================================================
# 3. Model Initialization
# ===================================================================
model = TransformerModel_add_embedding(
    ntoken=len(vocab), 
    d_model=config.embsize, 
    nhead=config.nheads,
    d_hid=config.d_hid, 
    nlayers=config.nlayers, 
    nlayers_cls=config.n_layers_cls,
    n_cls=1, 
    vocab=vocab, 
    dropout=config.dropout, 
    pad_token=config.pad_token,
    pad_value=config.pad_value, 
    do_mvc=True,
    use_batch_labels=False,
    explicit_zero_prob=False, 
    use_fast_transformer=True, 
    use_MVC_impute=True,
    impute_MVC_knn_k=config.impute_mvc_knn_k,
    esm_embedding_matrix=esm_matrix_for_model
)

base_params = []
esm_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if name.startswith("esm_encoder."):
        esm_params.append(param)
    else:
        base_params.append(param)

optimizer_grouped_parameters = [
    {"params": base_params, "lr": config.lr},
    {"params": esm_params, "lr": config.lr / 2.0},
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

model, optimizer = accelerator.prepare(model, optimizer)



# ===================================================================
# 4. Data Preparation
# ===================================================================
object_list_to_broadcast = [None, None, None]

if accelerator.is_main_process:
    print("\n" + "="*20 + " Preprocessing Data (Main Process) " + "="*20)
    all_files = sorted([p for p in Path(config.data_path).glob("*.h5ad")])
    random.seed(42)
    
    def prepare_data_and_split_patches(file_paths, val_split_ratio, patch_size):
        all_adatas = []
        train_patch_metas = []
        val_patch_metas = []
        
        print(f"Processing {len(file_paths)} files...")
        for adata_idx, file_path in enumerate(file_paths):
            try:
                adata = sc.read_h5ad(file_path)

                all_adatas.append(adata)
                
                # Spatial Indexing for Patch Generation
                coords_np = np.stack([adata.obs['cell_x'].values, adata.obs['cell_y'].values], axis=1)
                kdtree = cKDTree(coords_np)
                
                num_patches = adata.n_obs // patch_size
                if num_patches == 0:
                    continue
                    
                center_indices = np.random.choice(adata.n_obs, num_patches, replace=False)
                _, all_patch_indices_for_file = kdtree.query(coords_np[center_indices], k=patch_size)
                
                np.random.shuffle(all_patch_indices_for_file)
                split_point = int(len(all_patch_indices_for_file) * (1 - val_split_ratio))
                
                train_patches = all_patch_indices_for_file[:split_point]
                val_patches = all_patch_indices_for_file[split_point:]
                
                for p_indices in train_patches:
                    train_patch_metas.append((adata_idx, p_indices))
                for p_indices in val_patches:
                    val_patch_metas.append((adata_idx, p_indices))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return all_adatas, train_patch_metas, val_patch_metas

    all_adatas, train_patch_metas, val_patch_metas = prepare_data_and_split_patches(
        all_files, config.validation_split, config.patch_size
    )
    
    print(f"Total AnnData objects: {len(all_adatas)}")

    
    object_list_to_broadcast = [all_adatas, train_patch_metas, val_patch_metas]

accelerator.wait_for_everyone()
broadcast_object_list(object_list_to_broadcast, from_process=0)
all_adatas, train_patch_metas, val_patch_metas = object_list_to_broadcast
del object_list_to_broadcast
gc.collect()

# Datasets and Dataloaders
train_dataset = MicroenvironmentAwarePatchDataset(all_adatas, train_patch_metas, vocab, config)
val_dataset = MicroenvironmentAwarePatchDataset(all_adatas, val_patch_metas, vocab, config)

collate_fn = partial(
    microenvironment_collate_fn, 
    max_length=config.max_length, 
    pad_token_id=vocab[config.pad_token],
    pad_value=config.pad_value, 
    mask_value=config.mask_value,
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, 
    collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False, 
    collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True
)

train_dataloader, val_dataloader = accelerator.prepare(train_dataloader, val_dataloader)

# ===================================================================
# 5. Learning Rate Scheduler
# ===================================================================
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = config.epochs * num_update_steps_per_epoch
num_warmup_steps = int(num_training_steps * config.warmup_ratio)

def lr_lambda_cosine(current_step: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    decayed_factor = config.lr_end_factor + (1.0 - config.lr_end_factor) * cosine_decay
    return decayed_factor

lr_scheduler = LambdaLR(optimizer, lr_lambda_cosine)
lr_scheduler = accelerator.prepare(lr_scheduler)

# ===================================================================
# 6. Training Loop
# ===================================================================
best_val_loss = float('inf')

for epoch in range(config.epochs):
    if accelerator.is_main_process:
        print(f"\n{'='*20} Epoch {epoch+1}/{config.epochs} {'='*20}")
    
    # --- Training Phase ---
    model.train()
    epoch_stats = {'train': {'loss': 0.0, 'batches': 0}, 'val': {'loss': 0.0, 'batches': 0}}
    
    for batch in train_dataloader:
        if batch is None: continue
        
        genes = batch["genes"]
        expressions_truth = batch["expressions_truth"]
        expressions_masked = batch["expressions_masked"]
        coordinates = batch["coordinates"]
        mlm_mask = batch["mlm_mask"]
        patch_membership = batch["patch_membership"]
        origin_coordinates = batch["origin_coordinates"]
        
        src_key_padding_mask = (genes == vocab[config.pad_token])


        output = model(
            src=genes, 
            values=expressions_masked, 
            src_key_padding_mask=src_key_padding_mask,
            coordinates=coordinates, 
            CLS=False, 
            MVC=True, 
            ECS=False, 
            MVC_impute=True,
            generative_training=False
        )
        
        loss_mep = masked_mse_loss(output["mlm_output"], expressions_truth, mlm_mask)

        non_pad_mask = ~src_key_padding_mask
        loss_cpr = masked_mse_loss(output["mvc_output"], expressions_truth, non_pad_mask)

        loss_nei = masked_mse_loss(output["impute_pred"], expressions_truth, non_pad_mask)

        cell_embeddings = output["cell_emb"]
        loss_scl = spatially_aware_contrastive_loss(
            cell_embeddings, 
            patch_membership, 
            origin_coordinates, 
            temperature=config.scl_temperature, 
            k_negatives=config.scl_k_negatives
        )

        combined_loss = (
            config.loss_weight_mep * loss_mep + 
            config.loss_weight_cpr * loss_cpr + 
            config.loss_weight_nei * loss_nei + 
            config.loss_weight_scl * loss_scl
        )

        optimizer.zero_grad()
        accelerator.backward(combined_loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()

        total_loss_g = accelerator.gather(combined_loss).mean().item()
        epoch_stats['train']['loss'] += total_loss_g
        epoch_stats['train']['batches'] += 1

    # --- Validation Phase ---
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            if batch is None: continue
            
            genes = batch["genes"]
            expressions_truth = batch["expressions_truth"]
            expressions_masked = batch["expressions_masked"]
            coordinates = batch["coordinates"]
            mlm_mask = batch["mlm_mask"]
            patch_membership = batch["patch_membership"]
            origin_coordinates = batch["origin_coordinates"]
            src_key_padding_mask = (genes == vocab[config.pad_token])

            output = model(
                src=genes, values=expressions_masked, src_key_padding_mask=src_key_padding_mask,
                coordinates=coordinates, CLS=False, MVC=True, ECS=False, MVC_impute=True,
                generative_training=False
            )
            
            loss_mep = masked_mse_loss(output["mlm_output"], expressions_truth, mlm_mask)
            non_pad_mask = ~src_key_padding_mask
            loss_cpr = masked_mse_loss(output["mvc_output"], expressions_truth, non_pad_mask)
            loss_nei = masked_mse_loss(output["impute_pred"], expressions_truth, non_pad_mask)
            
            cell_embeddings = output["cell_emb"]
            loss_scl = spatially_aware_contrastive_loss(
                cell_embeddings, patch_membership, origin_coordinates, 
                temperature=config.scl_temperature, k_negatives=config.scl_k_negatives
            )
            
            combined_loss = (
                config.loss_weight_mep * loss_mep + 
                config.loss_weight_cpr * loss_cpr + 
                config.loss_weight_nei * loss_nei + 
                config.loss_weight_scl * loss_scl
            )

            total_loss_g = accelerator.gather(combined_loss).mean().item()
            epoch_stats['val']['loss'] += total_loss_g
            epoch_stats['val']['batches'] += 1

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        avg_val_loss = epoch_stats['val']['loss'] / max(1, epoch_stats['val']['batches'])
        
        print(f"Summary Epoch {epoch+1}:")
        print(f"  Val Loss:   {avg_val_loss:.4f}")

        unwrapped_model = accelerator.unwrap_model(model)
        
        torch.save(unwrapped_model.state_dict(), save_dir / f"model_epoch_{epoch+1}.pt")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(unwrapped_model.state_dict(), save_dir / "best_model.pt")
            print(f"  >> New best model saved (Loss: {best_val_loss:.4f})")

accelerator.end_training()
if accelerator.is_main_process:
    print("\nPre-training finished successfully!")