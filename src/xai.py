import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def grad_cam(model, img_array_01, last_conv_name, img_size, device='cuda'):
    """
    Generates a Grad-CAM overlay for a given image in PyTorch (no OpenCV required).
    """
    model.eval()
    model.to(device)

    # Convert numpy image (H, W, C) -> torch tensor (1, C, H, W)
    img_tensor = torch.from_numpy(img_array_01.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Hook to capture gradients and activations
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    # Register hooks on the chosen convolutional layer
    for name, module in model.named_modules():
        if name == last_conv_name:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)
            break
    else:
        raise ValueError(f"Layer {last_conv_name} not found in model")

    # Forward + backward passes
    outputs = model(img_tensor)
    class_idx = torch.argmax(outputs, dim=1).item()

    model.zero_grad()
    target = outputs[0, class_idx]
    target.backward()

    # Extract data
    act = activations['value'].detach()
    grad = gradients['value'].detach()

    # Global average pool gradients
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()

    # Normalize CAM to [0, 1]
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    # Resize heatmap with torch.interpolate instead of cv2
    cam_resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(img_size[0], img_size[1]),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()

    # Apply matplotlib colormap
    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(cam_resized)[..., :3]

    # Blend with the original image
    overlay = (img_array_01 * 0.6) + (heatmap_colored * 0.4)
    overlay = np.clip(overlay, 0, 1)

    return overlay
