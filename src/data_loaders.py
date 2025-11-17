import os
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.models import VGG16_Weights, ResNet50_Weights, Inception_V3_Weights


# -------------------------------------------------------------------
# Dataset wrapper that applies transforms to any base dataset or subset
# -------------------------------------------------------------------
class DatasetWithTransform(Dataset):
    """Wraps a PyTorch Dataset or Subset and applies a torchvision transform."""

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# -------------------------------------------------------------------
# Load CIFAR-10 and split into train / val / test datasets
# -------------------------------------------------------------------
def load_cifar10(val_ratio=0.1, seed=42):
    """Loads CIFAR-10 and creates PyTorch Dataset splits (train, val, test)."""

    # Load raw CIFAR-10 datasets without transforms yet
    full_train = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)

    # Validation split
    n_val = int(len(full_train) * val_ratio)
    n_train = len(full_train) - n_val
    train_dataset, val_dataset = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


# -------------------------------------------------------------------
# Create DataLoaders for model training / validation / testing
# -------------------------------------------------------------------
def make_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=64,
    model_name="vgg",
    augment=True,
):
    """Creates PyTorch DataLoaders with preprocessing and augmentation."""

    num_workers = os.cpu_count() // 2 if os.cpu_count() else 2
    image_size = (224, 224) if model_name in ("vgg", "resnet") else (299, 299)

    # Model-specific weights for normalization
    weights_map = {
        "vgg": VGG16_Weights.IMAGENET1K_V1,
        "resnet": ResNet50_Weights.IMAGENET1K_V1,
        "inception": Inception_V3_Weights.IMAGENET1K_V1,
    }
    if model_name not in weights_map:
        raise ValueError(f"Unsupported model choice: {model_name}")

    weights = weights_map[model_name]

    # ✅ TorchVision 0.20+ uses ImageClassification transforms, not Compose
    if hasattr(weights, "transforms"):
        preprocess = weights.transforms()
        # Try the new API
        if hasattr(preprocess, "mean") and hasattr(preprocess, "std"):
            mean, std = preprocess.mean, preprocess.std
        else:
            # Fallback to standard ImageNet normalization
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        # Legacy fallback (for <=0.19)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # Compose transforms
    base_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose(base_transforms)

    eval_transform = transforms.Compose(base_transforms)

    # Wrap datasets with transforms
    train_dataset = DatasetWithTransform(train_dataset, transform=train_transform)
    val_dataset = DatasetWithTransform(val_dataset, transform=eval_transform)
    test_dataset = DatasetWithTransform(test_dataset, transform=eval_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print(f"✅ mean={mean}, std={std}")
    return train_loader, val_loader, test_loader, mean, std
