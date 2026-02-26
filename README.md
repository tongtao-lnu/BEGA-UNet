\# EGA-UNet: Edge-Guided Attention U-Net for Colonoscopic Polyp Segmentation



\[!\[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

\[!\[PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-red.svg)](https://pytorch.org/)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



Official PyTorch implementation of the paper:



> \*\*EGA-UNet: Edge-Guided Attention U-Net with Multi-Scale Feature Aggregation for Colonoscopic Polyp Segmentation\*\*

>

> \[Paper Link (Coming Soon)]()



\## Abstract



Accurate polyp segmentation from colonoscopy images is critical for colorectal cancer prevention. We propose EGA-UNet, a boundary-aware segmentation architecture that introduces explicit edge modeling as a structural inductive prior. The framework integrates three components: an \*\*Edge-Guided Module (EGM)\*\* with learnable Sobel-initialized operators, a \*\*Dual-Path Attention (DPA)\*\* module, and a \*\*Multi-Scale Feature Aggregation (MSFA)\*\* module. EGA-UNet achieves \*\*88.53% Dice\*\* and \*\*82.51% IoU\*\* on the combined benchmark, and maintains \*\*83.2%\*\* of in-distribution performance under cross-dataset evaluation.



\## Architecture



!\[EGA-UNet Architecture](figures/architecture.png)



\## Results



\### In-Distribution Performance (Kvasir-SEG + CVC-ClinicDB)



| Model | Dice (%) | IoU (%) | HD95 ↓ |

|-------|----------|---------|--------|

| U-Net | 82.38 | 74.37 | 47.01 |

| Attention U-Net | 83.95 | 75.81 | 46.66 |

| TransUNet | 83.91 | 76.10 | 44.10 |

| \*\*EGA-UNet (Ours)\*\* | \*\*88.53\*\* | \*\*82.51\*\* | \*\*28.20\*\* |



\### Cross-Dataset Generalization



| Model | K→C Dice (%) | C→K Dice (%) | Retention |

|-------|--------------|--------------|-----------|

| U-Net | 54.70 | 51.49 | 64.5% |

| Attention U-Net | 44.59 | 35.20 | 47.5% |

| TransUNet | 50.37 | 38.70 | 53.1% |

| \*\*EGA-UNet (Ours)\*\* | \*\*70.33\*\* | \*\*77.04\*\* | \*\*83.2%\*\* |



\## Installation



\### Requirements

\- Python >= 3.8

\- PyTorch >= 2.0

\- CUDA >= 11.8 (for GPU training)



\### Setup



```bash

\# Clone the repository

git clone https://github.com/tongtao-lnu/EGA-UNet.git

cd EGA-UNet



\# Create conda environment (recommended)

conda create -n egaunet python=3.8

conda activate egaunet



\# Install dependencies

pip install -r requirements.txt



&nbsp;Dataset Preparation

1\. Download Datasets

Kvasir-SEG: Download

CVC-ClinicDB: Download

ETIS-Larib (for zero-shot): Download

2\. Organize Data Structure

data/

├── images/

│   ├── cju0qkwl35piu0993l0dewei2.jpg

│   ├── ...

└── masks/

&nbsp;   ├── cju0qkwl35piu0993l0dewei2.jpg

&nbsp;   └── ...



3\. Preprocess Data

python preprocess\_data.py



This will create processed\_data/ with train/val/test splits.

&nbsp;Training

Train EGA-UNet

python train.py



Training Configuration

Key hyperparameters (in train.py):

epochs: 100

batch\_size: 8

learning\_rate: 1e-4

image\_size: 352×352

Monitor Training

tensorboard --logdir=outputs/logs



&nbsp;Evaluation

Test Set Evaluation

python evaluate\_test.py



Cross-Dataset Evaluation

\# Prepare cross-dataset data

python cross\_dataset/prepare\_data.py



\# Run cross-dataset experiments

python cross\_dataset/run\_all\_experiments.py



Zero-Shot Evaluation (ETIS-Larib)

python zero\_shot\_evaluation.py



&nbsp;Pre-trained Weights

Pre-trained model weights are available:

Model	Dataset	Dice	Download

EGA-UNet	Kvasir+CVC	88.53%	Google Drive / Baidu Pan



Place the downloaded best\_model.pth in outputs/checkpoints/ega\_unet/.

&nbsp;Project Structure

EGA-UNet/

├── models/

│   ├── ega\_unet.py          # Main model architecture

│   ├── ablation\_models.py   # Ablation study variants

│   └── baselines/           # Comparison methods

├── utils/

│   ├── dataset.py           # Data loading

│   ├── losses.py            # Loss functions

│   └── metrics.py           # Evaluation metrics

├── cross\_dataset/           # Cross-dataset experiments

├── train.py                 # Training script

├── evaluate\_test.py         # Evaluation script

├── preprocess\_data.py       # Data preprocessing

└── zero\_shot\_evaluation.py  # Zero-shot evaluation



&nbsp;Ablation Study

To run ablation experiments:

\# Baseline (U-Net only)

python -c "from models.ablation\_models import get\_ablation\_model; m, c = get\_ablation\_model('baseline'); print(c)"



\# EGM only

python -c "from models.ablation\_models import get\_ablation\_model; m, c = get\_ablation\_model('egm\_only'); print(c)"



\# Full model

python -c "from models.ega\_unet import EGAUNet; m = EGAUNet(); print(sum(p.numel() for p in m.parameters())/1e6, 'M')"



&nbsp;Citation

If you find this work useful, please cite:

@article{author2024egaunet,

&nbsp; title={EGA-UNet: Edge-Guided Attention U-Net with Multi-Scale Feature Aggregation for Colonoscopic Polyp Segmentation},

&nbsp; author={Author Names},

&nbsp; journal={Journal Name},

&nbsp; year={2024}

}



&nbsp;License

This project is licensed under the MIT License - see the LICENSE file for details.

&nbsp;Acknowledgments

Kvasir-SEG Dataset

CVC-ClinicDB Dataset

ETIS-Larib Dataset

&nbsp;Contact

For questions, please open an issue or contact: \[your-email@example.com]

