import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import base64
import io
import os
import time
import requests
from config import MODEL_DIR, MODEL_PATH


os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_half = True if torch.cuda.is_available() else False
print("Using device:", device)

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

upsampler = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=512,
    tile_pad=10,
    pre_pad=0,
    half=use_half,
    device=device
    )

def b64_to_image(b64_string):
    decoded = base64.b64decode(b64_string)
    buffer = io.BytesIO(decoded)
    img = Image.open(buffer)
    if img.mode not in ['RGB', 'RGBA']:
        img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
    return img

def image_to_b64(img):
    buffer = io.BytesIO()
    if img.mode == 'RGBA':
        img.save(buffer, format="PNG")
    else:
        img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def upscale_b64(b64_image):
    img = b64_to_image(b64_image)
    if img.mode == 'RGBA':
        r, g, b, a = img.split()
        rgb_img = Image.merge('RGB', (r, g, b))
        rgb_np = np.array(rgb_img, dtype=np.uint8)
        upscaled_rgb, _ = upsampler.enhance(rgb_np, outscale=4)
        alpha_np = np.array(a, dtype=np.uint8)
        upscaled_alpha, _ = upsampler.enhance(alpha_np, outscale=4)
        upscaled_r, upscaled_g, upscaled_b = upscaled_rgb[:,:,0], upscaled_rgb[:,:,1], upscaled_rgb[:,:,2]
        out_img = Image.merge('RGBA', (
            Image.fromarray(upscaled_r),
            Image.fromarray(upscaled_g), 
            Image.fromarray(upscaled_b),
            Image.fromarray(upscaled_alpha[:,:,0] if len(upscaled_alpha.shape) == 3 else upscaled_alpha)
        ))
    else:
        img_np = np.array(img, dtype=np.uint8)
        output_np, _ = upsampler.enhance(img_np, outscale=4)
        out_img = Image.fromarray(output_np)

    os.makedirs("uploads", exist_ok=True)
    output_path = f"uploads/upscaled_{int(time.time())}.png"
    out_img.save(output_path)

    return {
        "file_path": output_path,
        "base64": image_to_b64(out_img)
    }

if __name__ == "__main__":
    with open("input.png", "rb") as f:
        b64_input = base64.b64encode(f.read()).decode()

    result = upscale_b64(b64_input)
    print("Saved:", result["file_path"])
    print(result["base64"][:200])
