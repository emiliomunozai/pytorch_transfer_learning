import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

num_classes = 10  # Adjust as needed

# ---- HEAD ----
def build_head(base_model, num_classes=num_classes, dropout=0.25):
    """
    Builds a classification head on top of a base model.
    Equivalent to the Keras version, but in PyTorch.
    """
    in_features = base_model.fc.in_features if hasattr(base_model, "fc") else base_model.classifier[-1].in_features
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
        nn.Softmax(dim=1)
    )
    return head


# ---- VGG16 ----
def build_vgg(image_size):
    base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    base.classifier = nn.Identity()  # remove top classifier
    for param in base.parameters():
        param.requires_grad = False

    # build head
    head = build_head(base, num_classes, dropout=0.30)
    model = nn.Sequential(base, head)
    last_conv_name = "features.29"  # VGG last conv layer index
    return model, base, last_conv_name


# ---- RESNET50 ----
def build_resnet(image_size):
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in base.parameters():
        param.requires_grad = False

    in_features = base.fc.in_features
    base.fc = nn.Identity()  # remove top classifier

    head = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, num_classes),
        nn.Softmax(dim=1)
    )

    model = nn.Sequential(base, head)
    last_conv_name = "layer4.2.conv3"
    return model, base, last_conv_name


# ---- INCEPTIONV3 ----
def build_inception(image_size=(299, 299), num_classes=10):
    
    base = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        aux_logits=True
    )

    # Freeze pretrained layers
    for param in base.parameters():
        param.requires_grad = False

    # Replace the final FC layer
    in_features = base.fc.in_features
    base.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    if base.aux_logits:
        in_features_aux = base.AuxLogits.fc.in_features
        base.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    return base
