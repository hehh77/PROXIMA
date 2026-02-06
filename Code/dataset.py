import torch
import torch.nn.functional as F
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
import pandas as pd
import gc

def build_protein_knowledge_embedding(esm_dict, vocab, esm_embedding_dim):


    embedding_matrix = torch.empty(len(vocab), esm_embedding_dim)
    torch.nn.init.normal_(embedding_matrix)  # Default random initialization

    found_count = 0
    for gene_name, gene_id in vocab.get_stoi().items():
        if gene_name in esm_dict:
            # Use pre-computed ESM embeddings
            embedding_matrix[gene_id] = torch.from_numpy(esm_dict[gene_name])
            found_count += 1
    
    # Normalize per row
    embedding_matrix = F.normalize(embedding_matrix, p=2, dim=1)
    
    return embedding_matrix

class MicroenvironmentAwarePatchDataset(Dataset):
    """
    Dataset implementation for Microenvironment-Aware Patch Sampling.
    Groups spatially proximal cells into patches to capture tissue architecture.
    """
    def __init__(self, adatas, patch_metas, vocab, config):
        self.adatas = adatas
        self.patch_metas = patch_metas
        self.vocab = vocab
        self.config = config

        # Pre-load expression matrices for efficiency
        self.expression_matrix_list = []
        for adata in self.adatas:
            self.expression_matrix_list.append(adata.X.tocsr())

        # Map gene names to vocabulary IDs
        self.gene_ids_in_vocab_list = []
        pad_token_id = vocab[config.pad_token]
        
        for adata in self.adatas:
            gene_ids = []
            # Assuming adata.var['gene_name'] holds the marker names
            for gene_name in adata.var['gene_name']:
                try:
                    gene_ids.append(vocab[gene_name])
                except KeyError:
                    # Fallback for genes not in vocab
                    gene_ids.append(pad_token_id)
            self.gene_ids_in_vocab_list.append(torch.tensor(gene_ids, dtype=torch.long))

    def __len__(self):
        return len(self.patch_metas)

    def __getitem__(self, idx):
        adata_idx, absolute_indices_for_patch = self.patch_metas[idx]
        
        # Retrieve the correct expression matrix
        expr_matrix = self.expression_matrix_list[adata_idx]
        gene_ids_in_vocab = self.gene_ids_in_vocab_list[adata_idx]
        
        # Retrieve spatial coordinates
        adata = self.adatas[adata_idx]
        all_coords = torch.from_numpy(
            np.stack([adata.obs['cell_x'].values, adata.obs['cell_y'].values], axis=1).astype(np.float32)
        )
        
        patch_genes_list, patch_exprs_list = [], []
        
        for cell_idx in absolute_indices_for_patch:
            row_expr_numpy = expr_matrix[cell_idx].toarray().flatten()
            row_expr = torch.from_numpy(row_expr_numpy).float()
            
            nonzero_indices = torch.nonzero(row_expr, as_tuple=True)[0]
            genes = gene_ids_in_vocab[nonzero_indices]
            expressions = row_expr[nonzero_indices]
            
            # Add CLS token and PAD value
            genes = torch.cat([torch.tensor([self.vocab[self.config.cls_token]]), genes])
            expressions = torch.cat([torch.tensor([self.config.pad_value]), expressions])
            
            patch_genes_list.append(genes)
            patch_exprs_list.append(expressions)
            
        patch_coords = all_coords[absolute_indices_for_patch]

        return {
            "genes": patch_genes_list, 
            "expressions": patch_exprs_list, 
            "coordinates": patch_coords,
            "absolute_indices": torch.tensor(absolute_indices_for_patch, dtype=torch.long),
            "adata_idx": adata_idx,
        }

def microenvironment_collate_fn(batch_of_patches, max_length, pad_token_id, pad_value, mlm_prob=0.15, mask_value=-1):
    """
    Collate function that handles padding, masking (for MEP), and coordinate offsets for patches.
    """
    unique_cells_data = {}

    for patch_idx, patch in enumerate(batch_of_patches):
        adata_idx = patch['adata_idx']  
        for i in range(len(patch['absolute_indices'])):
            cell_abs_idx = patch['absolute_indices'][i].item()
            unique_key = (adata_idx, cell_abs_idx)
            
            # De-duplication: process each cell only once per batch
            if unique_key not in unique_cells_data:
                unique_cells_data[unique_key] = {
                    "genes": patch['genes'][i],
                    "expressions": patch['expressions'][i],
                    "coordinates": patch['coordinates'][i],
                    "patch_id": patch_idx 
                }

    if not unique_cells_data: 
        return None

    genes_list = [data['genes'] for data in unique_cells_data.values()]
    expr_list = [data['expressions'] for data in unique_cells_data.values()]
    coords_list = [data['coordinates'] for data in unique_cells_data.values()]
    patch_membership_list = [data['patch_id'] for data in unique_cells_data.values()]

    # Truncation to max_length
    for i in range(len(genes_list)):
        if len(genes_list[i]) > max_length:
            cls_gene, cls_expr = genes_list[i][0], expr_list[i][0]
            indices = torch.randperm(len(genes_list[i]) - 1)[:max_length - 1] + 1
            genes_list[i] = torch.cat([cls_gene.unsqueeze(0), genes_list[i][indices]])
            expr_list[i] = torch.cat([cls_expr.unsqueeze(0), expr_list[i][indices]])
            
    # Padding
    current_max_len = max(len(g) for g in genes_list)
    padded_genes_list, padded_expr_list = [], []
    for i in range(len(genes_list)):
        pad_len = current_max_len - len(genes_list[i])
        genes_padded = torch.cat([genes_list[i], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        expr_padded = torch.cat([expr_list[i], torch.full((pad_len,), pad_value, dtype=torch.float32)])
        padded_genes_list.append(genes_padded)
        padded_expr_list.append(expr_padded)
        
    genes_tensor = torch.stack(padded_genes_list)
    expressions_tensor = torch.stack(padded_expr_list)

    origin_coordinates_tensor = torch.stack(coords_list)
    patch_ids_tensor = torch.tensor(patch_membership_list, dtype=torch.float32)
    
    # Offset coordinates to separate patches in spatial space (for attention mechanisms)
    MAX_COORD_OFFSET = 100000.0
    offset = patch_ids_tensor.unsqueeze(1) * MAX_COORD_OFFSET
    offset_coordinates = origin_coordinates_tensor.clone()
    offset_coordinates[:, 0] += offset.squeeze()

    # Mask Generation for MEP (Masked Expression Prediction)
    masked_expressions = expressions_tensor.clone()
    prob_matrix = torch.full(masked_expressions.shape, mlm_prob)
    prob_matrix[:, 0] = 0 # Do not mask CLS
    prob_matrix[genes_tensor == pad_token_id] = 0 # Do not mask PAD
    mlm_mask = torch.bernoulli(prob_matrix).bool()
    masked_expressions[mlm_mask] = mask_value

    patch_membership = torch.tensor(patch_membership_list, dtype=torch.long)

    return {
        "genes": genes_tensor, 
        "expressions_truth": expressions_tensor,
        "expressions_masked": masked_expressions, 
        "origin_coordinates": origin_coordinates_tensor,
        "coordinates": offset_coordinates,
        "mlm_mask": mlm_mask, 
        "patch_membership": patch_membership
    }





def prepare_annotation_data(config):
    """
    Loads and preprocesses data specifically for Cell Type Annotation tasks.
    """
    print("--- Loading and Preprocessing Data for Annotation ---")
    adata_train_val = sc.read_h5ad(config.train_val_h5ad_path)
    adata_test = sc.read_h5ad(config.test_h5ad_path)

    # Verify columns
    for adata, name in [(adata_train_val, "Train/Val"), (adata_test, "Test")]:
        if config.cell_type_key not in adata.obs.columns:
            raise ValueError(f"Key '{config.cell_type_key}' not found in {name} .obs")

    # Unified cell type mapping
    all_cell_types = pd.concat([
        adata_train_val.obs[config.cell_type_key],
        adata_test.obs[config.cell_type_key]
    ]).unique()
    
    cell_type_map = {name: i for i, name in enumerate(all_cell_types)}
    inverse_cell_type_map = {i: name for name, i in cell_type_map.items()}
    
    adata_train_val.obs["cell_type_id"] = adata_train_val.obs[config.cell_type_key].map(cell_type_map).astype(int)
    adata_test.obs["cell_type_id"] = adata_test.obs[config.cell_type_key].map(cell_type_map).astype(int)
    
    config.num_cell_types = len(cell_type_map)

    # Filter genes based on vocab
    from Src.tokenizer import GeneVocab # Local import to avoid circular dependency if needed
    vocab = GeneVocab.from_file(config.vocab_path)
    for s in config.special_tokens:
        if s not in vocab:
            vocab.append_token(s)
            
    processed_adatas = []
    for adata in [adata_train_val, adata_test]:
        adata.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata.var.index]
        genes_in_vocab_mask = adata.var["id_in_vocab"] >= 0
        adata = adata[:, genes_in_vocab_mask].copy()
        processed_adatas.append(adata)
    
    adata_train_val, adata_test = processed_adatas
    gc.collect()

    # Split Train/Val
    np.random.seed(config.seed) 
    indices = np.random.permutation(adata_train_val.n_obs)
    val_size = int(adata_train_val.n_obs * config.validation_split)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    
    return adata_train_val, adata_test, train_idx, val_idx, vocab, cell_type_map, inverse_cell_type_map

class PhenotypeAnnotationDataset(Dataset):
    """
    Dataset for supervised cell phenotype annotation.
    """
    def __init__(self, adata: sc.AnnData, indices: np.ndarray, vocab, config):
        self.adata = adata
        self.indices = indices
        self.vocab = vocab
        self.config = config
        
        self.expression_matrix = self.adata.X.tocsr() if hasattr(self.adata.X, 'tocsr') else self.adata.X
        self.gene_ids_in_vocab = torch.tensor(self.adata.var["id_in_vocab"].values, dtype=torch.long)
        self.cell_type_ids = torch.tensor(self.adata.obs["cell_type_id"].values, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cell_abs_idx = self.indices[idx]
        row_expr = self.expression_matrix[cell_abs_idx]
        if hasattr(row_expr, 'toarray'):
            row_expr = row_expr.toarray().flatten()
        
        row_expr = torch.from_numpy(row_expr).float()
        nonzero_indices = torch.nonzero(row_expr, as_tuple=True)[0]
        
        genes = self.gene_ids_in_vocab[nonzero_indices]
        expressions = row_expr[nonzero_indices]
        
        # Add CLS token
        genes = torch.cat([torch.tensor([self.vocab[self.config.cls_token]]), genes])
        expressions = torch.cat([torch.tensor([self.config.pad_value]), expressions])
        
        return {
            "genes": genes,
            "expressions": expressions,
            "cell_type_id": self.cell_type_ids[cell_abs_idx],
        }

def annotation_collate_fn(batch, config, vocab):
    """
    Collate function for training (includes random cropping and MEP masking).
    """
    genes_list = [item['genes'] for item in batch]
    expr_list = [item['expressions'] for item in batch]
    cell_type_ids_list = [item['cell_type_id'] for item in batch]

    # Random crop/pad
    for i in range(len(genes_list)):
        if len(genes_list[i]) > config.max_length:
            cls_gene, cls_expr = genes_list[i][0], expr_list[i][0]
            indices = torch.randperm(len(genes_list[i]) - 1)[:config.max_length - 1] + 1
            genes_list[i] = torch.cat([cls_gene.unsqueeze(0), genes_list[i][indices]])
            expr_list[i] = torch.cat([cls_expr.unsqueeze(0), expr_list[i][indices]])

    current_max_len = max(len(g) for g in genes_list)
    padded_genes_list, padded_expr_list = [], []
    for i in range(len(genes_list)):
        pad_len = current_max_len - len(genes_list[i])
        genes_padded = torch.cat([genes_list[i], torch.full((pad_len,), vocab[config.pad_token], dtype=torch.long)])
        expr_padded = torch.cat([expr_list[i], torch.full((pad_len,), config.pad_value, dtype=torch.float32)])
        padded_genes_list.append(genes_padded)
        padded_expr_list.append(expr_padded)
        
    genes_tensor = torch.stack(padded_genes_list)
    expressions_tensor = torch.stack(padded_expr_list)
    cell_type_ids_tensor = torch.stack(cell_type_ids_list)

    # MEP (Masked Expression Prediction) Masking
    masked_expressions = expressions_tensor.clone()
    prob_matrix = torch.full(masked_expressions.shape, 0.15)
    prob_matrix[:, 0] = 0
    prob_matrix[genes_tensor == vocab[config.pad_token]] = 0
    mlm_mask = torch.bernoulli(prob_matrix).bool()
    masked_expressions[mlm_mask] = config.mask_value

    return {
        "genes": genes_tensor,
        "expressions_truth": expressions_tensor,
        "expressions_masked": masked_expressions,
        "mlm_mask": mlm_mask,
        "cell_type_ids": cell_type_ids_tensor,
    }

def annotation_eval_collate_fn(batch, config, vocab):
    """
    Collate function for evaluation (deterministic truncation, no masking).
    """
    genes_list = [item['genes'] for item in batch]
    expr_list = [item['expressions'] for item in batch]
    cell_type_ids_list = [item['cell_type_id'] for item in batch]

    # Deterministic truncation
    for i in range(len(genes_list)):
        if len(genes_list[i]) > config.max_length:
            cls_gene, cls_expr = genes_list[i][0], expr_list[i][0]
            genes_to_keep = genes_list[i][1:config.max_length]
            expr_to_keep = expr_list[i][1:config.max_length]
            genes_list[i] = torch.cat([cls_gene.unsqueeze(0), genes_to_keep])
            expr_list[i] = torch.cat([cls_expr.unsqueeze(0), expr_to_keep])

    current_max_len = max(len(g) for g in genes_list)
    padded_genes_list, padded_expr_list = [], []
    for i in range(len(genes_list)):
        pad_len = current_max_len - len(genes_list[i])
        genes_padded = torch.cat([genes_list[i], torch.full((pad_len,), vocab[config.pad_token], dtype=torch.long)])
        expr_padded = torch.cat([expr_list[i], torch.full((pad_len,), config.pad_value, dtype=torch.float32)])
        padded_genes_list.append(genes_padded)
        padded_expr_list.append(expr_padded)
        
    genes_tensor = torch.stack(padded_genes_list)
    expressions_tensor = torch.stack(padded_expr_list)
    cell_type_ids_tensor = torch.stack(cell_type_ids_list)

    # No masking for eval
    masked_expressions = expressions_tensor.clone()
    mlm_mask = torch.zeros_like(masked_expressions, dtype=torch.bool)

    return {
        "genes": genes_tensor,
        "expressions_truth": expressions_tensor,
        "expressions_masked": masked_expressions,
        "mlm_mask": mlm_mask,
        "cell_type_ids": cell_type_ids_tensor,
    }


def prepare_imputation_data(config):
    """
    Loads and preprocesses data specifically for Imputation tasks.
    """
    print("--- Loading and Preprocessing Data for Imputation ---")
    adata_tv = sc.read_h5ad(config.train_val_h5ad_path)
    
    # Load Vocab
    from Src.tokenizer import GeneVocab
    vocab = GeneVocab.from_file(config.vocab_path)
    
    # Map genes to vocab IDs
    adata_tv.var["id_in_vocab"] = [vocab[g] if g in vocab else -1 for g in adata_tv.var.index]
    genes_in_vocab_mask = adata_tv.var["id_in_vocab"] >= 0
    adata_tv = adata_tv[:, genes_in_vocab_mask].copy()
    
    # Split Train/Val
    np.random.seed(config.seed)
    indices = np.random.permutation(adata_tv.n_obs)
    val_size = int(adata_tv.n_obs * config.validation_split)
    val_idx, train_idx = indices[:val_size], indices[val_size:]
    
    return adata_tv, train_idx, val_idx, vocab

class ProximaImputationDataset(Dataset):
    """
    Dataset for Imputation. Returns full expression profiles.
    Masking is handled in the collate function.
    """
    def __init__(self, adata: sc.AnnData, indices: np.ndarray):
        self.adata = adata
        self.indices = indices
        self.expression_matrix = adata.X.tocsr() if hasattr(adata.X, 'tocsr') else adata.X
        self.gene_ids_in_vocab = torch.tensor(self.adata.var["id_in_vocab"].values, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        cell_abs_idx = self.indices[idx]
        row_expr = self.expression_matrix[cell_abs_idx]
        if hasattr(row_expr, 'toarray'):
            row_expr = row_expr.toarray().flatten()
            
        row_expr = torch.from_numpy(row_expr).float()
        
        return {
            "expressions": row_expr, 
            "genes": self.gene_ids_in_vocab
        }

def imputation_collate_fn(batch, config, vocab):
    """
    Collate function for Imputation.
    Performs 'Hard Masking': randomly masks a percentage of NON-ZERO values 
    to simulate technical dropout and force the model to reconstruct them.
    """
    expr_list = [item['expressions'] for item in batch]
    genes_list = [item['genes'] for item in batch]
    
    expressions_truth = torch.stack(expr_list)
    genes_tensor = torch.stack(genes_list)


    expressions_masked_input = expressions_truth.clone()

    non_zero_mask = expressions_masked_input > 0

    prob_matrix = torch.full(expressions_masked_input.shape, config.imputation_masking_ratio)
    
    imputation_eval_mask = torch.bernoulli(prob_matrix).bool() & non_zero_mask
    
    expressions_masked_input[imputation_eval_mask] = 0
    
    cls_gene_id = torch.tensor([vocab[config.cls_token]])
    cls_expr_value = torch.tensor([config.pad_value])
    
    batch_size = expressions_truth.shape[0]
    

    genes_with_cls = torch.cat([cls_gene_id.expand(batch_size, -1), genes_tensor], dim=1)
    expressions_truth_with_cls = torch.cat([cls_expr_value.expand(batch_size, -1), expressions_truth], dim=1)
    expressions_masked_input_with_cls = torch.cat([cls_expr_value.expand(batch_size, -1), expressions_masked_input], dim=1)
    

    imputation_eval_mask_with_cls = F.pad(imputation_eval_mask, (1, 0), "constant", 0)

    return {
        "genes": genes_with_cls,
        "expressions_truth": expressions_truth_with_cls,
        "expressions_masked_input": expressions_masked_input_with_cls,
        "imputation_eval_mask": imputation_eval_mask_with_cls,
    }