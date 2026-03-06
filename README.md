# BEGA-UNet: Boundary-Explicit Guided Attention U-Net for Colonoscopic Polyp Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the paper:

> **BEGA-UNet: Boundary-Explicit Guided Attention U-Net with Multi-Scale Feature Aggregation for Colonoscopic Polyp Segmentation**
>
> Tao Tong, Wen Zhang, Wanni Zu*
>
> [https://doi.org/10.64898/2026.03.04.26347608]()

## Highlights

- **Explicit boundary modeling** via learnable Sobel-initialized Edge-Guided Module (EGM)
- **83.2% cross-domain performance retention**, substantially exceeding U-Net (64.5%), Attention U-Net (47.5%), and TransUNet (53.1%)
- **88.53% Dice** and **82.51% IoU** on combined Kvasir-SEG + CVC-ClinicDB benchmark
- Edge features exhibit **11.7× less cross-domain divergence** than appearance-based representations

## Architecture
BEGA-UNet integrates three complementary modules within an encoder-decoder framework:

- **Edge-Guided Module (EGM)**: Learnable Sobel-initialized operators for explicit boundary extraction
- **Dual-Path Attention (DPA)**: Parallel channel and spatial attention to preserve boundary signals
- **Multi-Scale Feature Aggregation (MSFA)**: Multi-scale context encoding at the bottleneck

![BEGA-UNet Architecture](architecture.png)

## Results

### In-Distribution Performance (Kvasir-SEG + CVC-ClinicDB)

| Model | Dice (%) | IoU (%) | HD95 |
| :-- | :-- | :-- | :-- |
| U-Net | 82.38 | 74.37 | 47.01 |
| Attention U-Net | 83.95 | 75.81 | 46.66 |
| TransUNet | 83.91 | 76.10 | 44.10 |
| **BEGA-UNet (Ours)** | **88.53** | **82.51** | **28.20** |

### Cross-Dataset Generalization

| Model | K to C Dice (%) | C to K Dice (%) | Retention |
| :-- | :-- | :-- | :-- |
| U-Net | 54.70 | 51.49 | 64.5% |
| Attention U-Net | 44.59 | 35.20 | 47.5% |
| TransUNet | 50.37 | 38.70 | 53.1% |
| **BEGA-UNet (Ours)** | **70.33** | **77.04** | **83.2%** |

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (for GPU training)


### Setup

```bash
# Clone the repository
git clone https://github.com/tongtao-lnu/BEGA-UNet.git
cd BEGA-UNet

# Create conda environment (recommended)
conda create -n egaunet python=3.8
conda activate egaunet

# Install dependencies
pip install -r requirements.txt
```


## Dataset Preparation

### 1. Download Datasets

- **Kvasir-SEG**: https://datasets.simula.no/kvasir-seg/
- **CVC-ClinicDB**: https://polyp.grand-challenge.org/CVCClinicDB/
- **ETIS-Larib** (for zero-shot evaluation): https://polyp.grand-challenge.org/EtisLarib/


### 2. Organize Data Structure

```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```


### 3. Preprocess Data

```bash
python preprocess_data.py
```

This will create `processed_data/` with train/val/test splits.

## Training

### Train EGA-UNet

```bash
python train.py
```


### Training Configuration

Key hyperparameters (in `train.py`):

- epochs: 100
- batch_size: 8
- learning_rate: 1e-4
- image_size: 352 x 352


### Monitor Training

```bash
tensorboard --logdir=outputs/logs
```


## Evaluation

### Test Set Evaluation

```bash
python evaluate_test.py
```


### Cross-Dataset Evaluation

```bash
# Prepare cross-dataset data
python cross_dataset/prepare_data.py

# Run cross-dataset experiments
python cross_dataset/run_all_experiments.py
```


### Zero-Shot Evaluation (ETIS-Larib)

```bash
python zero_shot_evaluation.py
```


## Pre-trained Weights

Pre-trained weights can be obtained by running the training script:

```bash
python train.py
```

Training takes approximately 2-3 hours on a single NVIDIA RTX 4060 GPU.

## Project Structure

```
BEGA-UNet/
├── models/
│   ├── ega_unet.py          # Main model architecture
│   ├── ablation_models.py   # Ablation study variants
│   └── baselines/           # Comparison methods
├── utils/
│   ├── dataset.py           # Data loading
│   ├── losses.py            # Loss functions
│   └── metrics.py           # Evaluation metrics
├── cross_dataset/           # Cross-dataset experiments
├── train.py                 # Training script
├── evaluate_test.py         # Evaluation script
├── preprocess_data.py       # Data preprocessing
└── zero_shot_evaluation.py  # Zero-shot evaluation
```


## Ablation Study

To run ablation experiments:

```bash
# Baseline (U-Net only)
python -c "from models.ablation_models import get_ablation_model; m, c = get_ablation_model('baseline'); print(c)"

# EGM only
python -c "from models.ablation_models import get_ablation_model; m, c = get_ablation_model('egm_only'); print(c)"

# Full model
python -c "from models.ega_unet import EGAUNet; m = EGAUNet(); print(sum(p.numel() for p in m.parameters())/1e6, 'M')"
```


## Citation

If you find this work useful, please cite:

```bibtex
@article{tong2026egaunet,
  title={BEGA-UNet: Boundary-Explicit Guided Attention U-Net with Multi-Scale Feature Aggregation for Colonoscopic Polyp Segmentation},
  author={Tong, Tao},
  year={2026}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)
- [CVC-ClinicDB Dataset](https://polyp.grand-challenge.org/CVCClinicDB/)
- [ETIS-Larib Dataset](https://polyp.grand-challenge.org/EtisLarib/)


## Contact

For questions, please open an issue or contact: 3695473134@qq.com

```

```



