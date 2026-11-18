# LPNDNet: Lightweight Pulmonary Nodule Detection Model

**LPNDNet: Lightweight Pulmonary Nodule Detection Model with Re-parameterized Hierarchical Abstraction and Dynamic Bidirectional Feature Fusion**

## Overview

LPNDNet is a lightweight pulmonary nodule detection model built upon YOLOv11n. It is designed to improve detection performance while reducing model complexity and computational cost.

The framework contains four main improvements:

1. **RepAHGNetV2 backbone:** A hierarchical feature extraction network built on HGNetV2, incorporating re-parameterized convolution (RepConv) and Efficient Channel Attention (ECA) to enhance feature representation with low inference overhead.
2. **DDB-HSFPN feature fusion:** A lightweight, bidirectional feature pyramid that combines semantic feature selection, bottom-up information aggregation, and DySample-based dynamic upsampling for multi-scale feature reconstruction.
3. **GLSC3k2 module:** An enhanced C3k2 design using a Global–Local State-space Block (GLSBlock) to jointly model global contextual relationships and local spatial details.
4. **Inner-ShapeIoU loss:** A bounding-box regression objective combining shape-aware geometric weighting with auxiliary-box-based IoU optimization to improve nodule localization.

The current research framework operates on **2D axial CT slices** derived from volumetric CT scans; it is not an end-to-end 3D nodule detection system.

## Repository Contents

```text
.
├── DySample.py          # Dynamic upsampling component
├── Inner-ShapeIoU.py    # Bounding-box regression loss implementation
├── LPNDNet.yaml         # Model architecture configuration
├── RepConv.py           # Re-parameterized convolution component
├── ECA.py               # Efficient Channel Attention component
├── train.py             # Training entry script
├── val.py               # Validation entry script
├── requirements.txt     # Python dependencies
├── LPNDNet.pt           # Trained weights of the proposed LPNDNet model
└── README.md            # Project documentation
```

## Environment

The experiments described in the manuscript used the following environment:

| Component | Configuration              |
| --------- | -------------------------- |
| Python    | 3.10.14                    |
| PyTorch   | 2.4.1                      |
| CUDA      | 12.1                       |
| GPU       | NVIDIA RTX 3080 Ti (12 GB) |
| CPU       | Intel Xeon Silver 4214R    |

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The experiments are conducted on the publicly available LUNA16 dataset and LIDC-IDRI dataset.

LUNA16: https://luna16.grand-challenge.org/

LIDC-IDRI: https://www.cancerimagingarchive.net/collection/lidc-idri/

The original CT scans are preprocessed into 2D images with corresponding bounding box annotations in YOLO format. The dataset is divided into training, validation, and testing subsets at the patient level using an 8:1:1 ratio.
Before training, prepare the dataset configuration file (data.yaml) and specify the corresponding dataset paths.

## Training and Validation

The principal settings for the LUNA16 experiments were:

| Setting                    | Value                                     |
| -------------------------- | ----------------------------------------- |
| Input image size           | 640 × 640                                 |
| Epochs                     | 300                                       |
| Batch size                 | 4                                         |
| Optimizer                  | AdamW                                     |
| Initial learning rate      | 0.002                                     |
| Final learning-rate factor | 0.01                                      |
| Weight decay               | 0.0005                                    |
| Momentum                   | 0.9                                       |
| Learning-rate schedule     | Linear decay                              |
| Warm-up epochs             | 3                                         |
| Mosaic augmentation        | 1.0 (disabled during the final 10 epochs) |

All models are trained from scratch without pretrained weights.

train.py` and `val.py` are provided as the training and validation entry scripts. 

The corresponding entry-point commands are:

```bash
python train.py
python val.py
```

## Experimental Results

Experiments on the LUNA16 dataset demonstrate that LPNDNet achieves improved pulmonary nodule detection performance with reduced model complexity.

Compared with the baseline YOLOv11 model, LPNDNet achieves:

- **7.03 percentage points** improvement in mAP50.
- **3.15 percentage points** improvement in mAP50-95.
- **40.31%** reduction in model parameters.

