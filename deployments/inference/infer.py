# inference.py
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from model import get_model


# -----------------------------
# Model loading
# -----------------------------
def load_model(model_path: str, device: str = "cpu"):
    model = get_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(image: Image.Image, max_dim: int = 350):
    h, w = image.size
    if h > w:
        new_h = max_dim
        new_w = int(max_dim * (w / h))
    else:
        new_w = max_dim
        new_h = int(max_dim * (h / w))

    new_h = 16 * (new_h // 16)
    new_w = 16 * (new_w // 16)

    resized = image.resize((new_h, new_w), Image.LANCZOS)

    lab = rgb2lab(np.array(resized) / 255.0)
    lab[..., 0] /= 100
    lab[..., 1:] = (lab[..., 1:] + 128) / 256

    L = torch.from_numpy(lab[..., 0]).float()
    lab_tensor = torch.from_numpy(lab).permute(2, 0, 1).float()

    return (
        L.unsqueeze(0),              # (1, H, W)
        lab_tensor.unsqueeze(0)      # (1, 3, H, W)
    )


def preprocess_palette(palette_rgb):
    """
    palette_rgb: List[List[int]] → [[R,G,B], ...] length = 6
    """
    palette_img = np.zeros((4, 24, 3), dtype=np.float32)

    for i, color in enumerate(palette_rgb):
        r = (i // 6) * 4
        c = (i % 6) * 4

        lab = rgb2lab(np.array(color).reshape(1, 1, 3) / 255.0).flatten()
        lab[0] /= 100
        lab[1:] = (lab[1:] + 128) / 256

        palette_img[r:r+4, c:c+4] = lab

    return torch.from_numpy(palette_img).permute(2, 0, 1).unsqueeze(0)


# -----------------------------
# Post-processing
# -----------------------------
def postprocess(output: torch.Tensor, original_size):
    out = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

    out[..., 0] = np.clip(out[..., 0], 0, 1) * 100
    out[..., 1:] = np.clip(out[..., 1:], 0, 1) * 255 - 128

    rgb = np.clip(lab2rgb(out) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb).resize(original_size, Image.LANCZOS)


# -----------------------------
# End-to-end inference
# -----------------------------
@torch.no_grad()
def recolor_image(
    model,
    image: Image.Image,
    palette_rgb,
    device: str = "cpu"
):
    L, src_lab = preprocess_image(image)
    palette = preprocess_palette(palette_rgb)

    L = L.to(device)
    src_lab = src_lab.to(device)
    palette = palette.to(device)

    output = model(src_lab, palette, L)
    return postprocess(output, image.size)
