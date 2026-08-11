import random
import warnings
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix as sk_contingency_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(y_true, y_pred):
    """
    Calculates classification metrics: Accuracy, Precision, Recall, and F1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def calculate_clustering_metrics(embeddings, cell_type_labels, full=False, seed=42):
    """
    Calculates clustering quality metrics based on cell type labels.
    """


    # Normalize and optionally reduce dimensionality
    emb_scaled = StandardScaler().fit_transform(embeddings)
    n_components = min(50, emb_scaled.shape[1])
    if emb_scaled.shape[1] > n_components:
        emb_reduced = PCA(n_components=n_components, random_state=seed).fit_transform(emb_scaled)
    else:
        emb_reduced = emb_scaled

    metrics = {}
    metrics["ASW_celltype"] = round(float(silhouette_score(emb_reduced, cell_type_labels)), 4)

    if full:
        n_clusters = len(np.unique(cell_type_labels))
        cluster_labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(emb_reduced)

        metrics["ARI_celltype"] = round(float(adjusted_rand_score(cell_type_labels, cluster_labels)), 4)
        metrics["NMI_celltype"] = round(float(normalized_mutual_info_score(cell_type_labels, cluster_labels)), 4)

        cm = sk_contingency_matrix(cell_type_labels, cluster_labels)
        metrics["PS_celltype"] = round(float(np.sum(np.amax(cm, axis=0)) / np.sum(cm)), 4)

    return metrics