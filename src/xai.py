import torch
import torch.nn.functional as F
import numpy as np


def run_forward(model, x):
    """
    Returns:
        output: logits tensor (1,C)
        pred:   python int
    """
    model.eval()
    if hasattr(model, "aux_logits"):
        model.aux_logits = False

    output = model(x)
    pred = output.argmax(dim=1).item()
    return output, pred

def attach_hooks(model, layer_name):
    """
    Returns:
        remove(): a function to detach hooks
        capture:  dict with 'act' and 'grad'
    """
    capture = {"act": None, "grad": None}

    def save_act(m, i, o):
        capture["act"] = o

    def save_grad(m, gi, go):
        capture["grad"] = go[0]

    target = dict(model.named_modules())[layer_name]

    h1 = target.register_forward_hook(save_act)
    h2 = target.register_backward_hook(save_grad)

    def remove():
        h1.remove()
        h2.remove()

    return capture, remove

def run_backward(output, pred_class):
    """
    Backprop for the predicted class.
    """
    output[:, pred_class].sum().backward()

def compute_gradcam(activations, gradients):
    """
    activations: (1, C, H, W)
    gradients:   (1, C, H, W)

    Returns:
        cam: torch tensor (H, W), float32, normalized [0,1]
    """
    C, H, W = activations.shape[1:]

    # GAP → (C,)
    weights = gradients[0].mean(dim=(1, 2))  

    # Weighted sum → (H, W)
    cam = (weights[:, None, None] * activations[0]).sum(dim=0)

    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam.float()

import torch.nn.functional as F

def resize_cam(cam, H, W):
    """
    cam: (Hc, Wc); torch float32
    Returns: resized (H, W)
    """
    cam_resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )[0, 0]
    return cam_resized


def tensor_to_uint8(img_t):
    """
    img_t: (3,H,W) torch tensor
    Returns: (H,W,3) uint8 numpy
    """
    img_t = img_t.detach().cpu()

    mn, mx = img_t.min(), img_t.max()
    if mx > mn:
        img_t = (img_t - mn) / (mx - mn)
    else:
        img_t = torch.zeros_like(img_t)

    arr = (img_t.permute(1,2,0).numpy() * 255).astype(np.uint8)
    return arr

def cam_to_uint8(cam):
    """
    cam: (H,W) float32 torch
    Returns (H,W) uint8 numpy
    """
    cam_np = (cam.detach().cpu().numpy() * 255).astype(np.uint8)
    return cam_np
