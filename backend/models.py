import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageSequence

from py_real_esrgan.rrdbnet_arch import RRDBNet
from py_real_esrgan.utils import (
    pad_reflect,
    split_image_into_overlapping_patches,
    stich_together,
    unpad_image,
)

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

HF_MODELS = {
    2: dict(repo_id="sberbank-ai/Real-ESRGAN", filename="RealESRGAN_x2.pth"),
    4: dict(repo_id="sberbank-ai/Real-ESRGAN", filename="RealESRGAN_x4.pth"),
}

device = torch.device("cpu")
_models: dict[int, "ESRGANModel"] = {}


class ESRGANModel:
    def __init__(self, scale: int):
        self.scale = scale
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        self._load_weights()
        self.model.eval()
        self.model.to(device)

    def _load_weights(self):
        WEIGHTS_DIR.mkdir(exist_ok=True)
        weight_path = WEIGHTS_DIR / HF_MODELS[self.scale]["filename"]
        if not weight_path.exists():
            cfg = HF_MODELS[self.scale]
            downloaded = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"], local_dir=str(WEIGHTS_DIR))
            if Path(downloaded) != weight_path:
                os.replace(downloaded, weight_path)
        loadnet = torch.load(str(weight_path), map_location=device, weights_only=False)
        if "params" in loadnet:
            self.model.load_state_dict(loadnet["params"], strict=True)
        elif "params_ema" in loadnet:
            self.model.load_state_dict(loadnet["params_ema"], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)

    def predict(self, lr_image: Image.Image, batch_size=4, patches_size=192, padding=24, pad_size=15) -> Image.Image:
        scale = self.scale
        lr_image_np = np.array(lr_image)
        lr_image_np = pad_reflect(lr_image_np, pad_size)
        patches, p_shape = split_image_into_overlapping_patches(lr_image_np, patch_size=patches_size, padding_size=padding)
        img = torch.FloatTensor(patches / 255).permute((0, 3, 1, 2)).to(device).detach()
        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i + batch_size])), 0)
        sr_image = res.permute((0, 2, 3, 1)).clamp_(0, 1).cpu().numpy()
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image_np.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(sr_image, padded_image_shape=padded_size_scaled, target_shape=scaled_image_shape, padding_size=padding * scale)
        sr_img = (np_sr_image * 255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size * scale)
        return Image.fromarray(sr_img)


def _get_model(scale: int) -> ESRGANModel:
    if scale not in _models:
        _models[scale] = ESRGANModel(scale)
    return _models[scale]


def upscale(input_path: str, output_path: str, scale: int) -> None:
    model = _get_model(2 if scale == 1 else scale)
    image = Image.open(input_path).convert("RGB")
    sr_image = model.predict(image)
    if scale == 1:
        sr_image = sr_image.resize(image.size, Image.LANCZOS)
    sr_image.save(output_path)


def upscale_gif(input_path: str, output_path: str, scale: int, progress_callback=None) -> None:
    model = _get_model(2 if scale == 1 else scale)
    gif = Image.open(input_path)
    loop = gif.info.get("loop", 0)

    # Must copy each frame immediately - Pillow reuses the same object
    frames_input = []
    durations = []
    for frame in ImageSequence.Iterator(gif):
        frames_input.append(frame.copy())
        durations.append(frame.info.get("duration", 100))

    total = len(frames_input)
    result_frames = []

    for i, frame in enumerate(frames_input):
        rgba = frame.convert("RGBA")
        rgb = rgba.convert("RGB")
        alpha = rgba.split()[3]

        sr_rgb = model.predict(rgb)
        target_size = sr_rgb.size if scale != 1 else rgb.size
        sr_alpha = alpha.resize(target_size, Image.BICUBIC)

        if scale == 1:
            sr_rgb = sr_rgb.resize(rgb.size, Image.LANCZOS)

        sr_rgba = sr_rgb.copy()
        sr_rgba.putalpha(sr_alpha)
        result_frames.append(sr_rgba)

        if progress_callback:
            progress_callback(i + 1, total)

    # Save as animated GIF
    result_frames[0].save(
        output_path,
        save_all=True,
        append_images=result_frames[1:],
        duration=durations,
        loop=loop,
        disposal=2,
    )
