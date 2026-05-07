# Transfer Learning Workshop

Comparative study of transfer learning using three pretrained ImageNet models — **VGG16**, **ResNet50**, and **InceptionV3** — fine-tuned on **CIFAR-10** (10-class image classification).

All backbone weights are frozen; only a lightweight classification head is trained on top. The project also includes a **Grad-CAM** explainability module to visualize which regions of an image the model focuses on when making predictions.

## Project structure

```
src/
  models.py         # Build VGG16, ResNet50, InceptionV3 with custom heads
  data_loaders.py   # CIFAR-10 loading, train/val/test split, augmentation
  model_training.py # Training loop, evaluation, random hyperparameter search
  xai.py            # Grad-CAM implementation (hooks, backward pass, heatmap)
transfer_learning_workshop.ipynb  # Main notebook
```

## Key features

- **Feature extraction**: backbone frozen, only the head is trained
- **Data augmentation**: random horizontal flip, rotation, affine transforms
- **Hyperparameter search**: random search over learning rate, batch size, optimizer, and regularization (L1 / L2 / none) for InceptionV3
- **Explainability**: Grad-CAM heatmaps overlaid on input images

## Setup

### GPU (CUDA 12.8)

```bash
uv sync --group gpu
```

### CPU only

```bash
uv sync --group cpu
```

Then open the notebook:

```bash
uv run jupyter notebook transfer_learning_workshop.ipynb
```
