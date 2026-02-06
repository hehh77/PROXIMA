# PROXIMA

## A microenvironment-aware foundation model for single-cell spatial proteomics

![Alt text](./fig1.png "PROXIMA Framework")

**PROXIMA** is a deep learning framework designed to decipher protein semantics and 3D tissue architecture.

## Hardware Requirements
PROXIMA is a large-scale foundation model designed for high-performance computing environments. It requires a standard workstation or server with sufficient RAM and multiple high-end NVIDIA GPUs to support distributed training and inference.

We validated the framework using the following specifications:
+ **CPU:** 64 cores, 2.90 GHz/core
+ **RAM:** 64 GB
+ **GPU:** 8 x NVIDIA RTX A6000

## System Requirements
This tool is supported for Linux. The tool has been tested on the following system:

+ Debian Linux 6.1.94-1 (kernel version 6.1.0-22) with x86_64 architecture

## Installation
To install the required packages for running PROXIMA, please use the following command:
```bash
conda create -n <env_name> python==3.9
conda activate <env_name>
pip install -r requirements.txt
```

### Time Cost
Typical installation time on a standard desktop computer is approximately 30 minutes, depending on internet speed for downloading PyTorch and CUDA dependencies.

## Usage
The utilization of PROXIMA involves two stages: **Pre-training** on large-scale corpora and **Fine-tuning** on downstream tasks. 

**Note:** All commands below should be executed within the `Code` directory.

```bash
cd ./Code
```

## 1. Pre-training

To launch distributed pre-training, use the `accelerate` library. Adjust the `num_processes` argument to match your available GPU count.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes=4 --mixed_precision=fp16 pretraining.py
```

### Time cost
Expected run time for demo on a "normal" desktop computer is about 100 minutes. This duration may vary depending on the hardware specifications and the size of the dataset.

### Expected Output
Upon successful execution, the script will create a directory structure under `../Results/PROXIMA_Pretraining`. The directory will be named with a timestamp (e.g., `20260108_142617`).

This directory will contain:
- `vocab.json`: The gene vocabulary used during training.
- `model_epoch_{N}.pt`: Checkpoints saved at each epoch.
- `best_model.pt`: The weights of the model with the lowest validation loss.

## 2. Fine-tuning: Cell Type Annotation

### Configuration
Before running the annotation task, you must update the configuration to point to your pre-trained model. 

Please open `finetuning_annotation.py` and replace the `pretrained_model_path` with the absolute path to the best checkpoint generated in the pre-training step.
In detail: please replace the "pretrained_model_path" using `../Results/PROXIMA_Pretraining/XXXXXXXX_XXXXXX/best_model.pt`, where `XXXXXXXX_XXXXXX` is the timestamp of the training.



### Running the Script
```bash
python finetuning_annotation.py
```

### Time cost
Expected run time for demo on a "normal" desktop computer is about 30 minutes. This duration may vary depending on the hardware specifications and the size of the dataset.

### Expected Output
Results will be saved in `../results_finetune/PROXIMA_Annotation_{TaskName}/{timestamp}`:
- `best_model_finetuned.pt`: The fine-tuned model weights.
- `test_predictions.csv`: A CSV file containing cell IDs, ground truth labels, and predicted cell types.
- `test_metrics.csv`: Final evaluation metrics (Accuracy, Macro F1, etc.).

## 3. Fine-tuning: Imputation
This task leverages PROXIMA to recover missing biological signals and perform in silico panel expansion.

### Configuration
Similar to the annotation task, you must update the pre-trained model path before running the script. Open `finetuning_imputation.py` and ensure `pretrained_model_path` points to your pre-trained checkpoint (e.g., `../Results/PROXIMA_Pretraining/XXXXXXXX_XXXXXX/best_model.pt`).

### Running the Script
```bash
python finetuning_imputation.py
```

### Time cost
Expected run time for demo on a "normal" desktop computer is about 60 minutes. This duration may vary depending on the hardware specifications and the size of the dataset.

### Expected Output
Results will be saved in `../results_finetune/PROXIMA_Imputation_{TaskName}/{timestamp}`:
- `best_model_imputation.pt`: The fine-tuned model optimized for reconstruction.
- Logs displaying the Validation Loss and Validation MAE.

## Data & Model Availability
Due to storage limitations, we provide a Demo Dataset for pre-training and downstream tasks, which is available on Zenodo: [https://doi.org/10.5281/zenodo.18501404]. Please place files in the `./Data` directory for pre-training and downstream tasks.

The Pre-training dataset was constructed using data sourced from the [Aquila database](https://aquila.cheunglab.org/) and additional datasets accessible at Zenodo (DOI: [10.5281/zenodo.10067009](https://doi.org/10.5281/zenodo.10067009)).
The complete, processed pre-training dataset and the associated pre-trained PROXIMA weights will be made publicly available upon publication.







