import numpy as np
import matplotlib.pyplot as plt

# Undo normalization for visualization
def unnormalize(img, mean, std):
    img = img.numpy().transpose((1, 2, 0))  # CHW -> HWC
    img = img * std + mean  # Unnormalize
    img = np.clip(img, 0, 1)
    return img

def display_images(images, labels, mean, std):
# Show a grid of images
    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axs.flatten()):
        if i >= images.size(0): break
        img = unnormalize(images[i], np.array(mean), np.array(std))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Label: {labels[i].item()}")
    plt.tight_layout()
    plt.show()