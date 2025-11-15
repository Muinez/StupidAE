
# StupidAE — d8c16 Tiny Patch Autoencoder

StupidAE is a very small, very fast, and intentionally simple model that still works surprisingly well.  
It has **13.24M parameters**, compresses by **8× per spatial dimension**, and uses **16 latent channels**.

The main goal: make a AE that doesn’t slow everything down and is fast enough to run directly during text-to-image training.

---

## Weights

The weights are available on HuggingFace:

👉 [https://huggingface.co/Muinez/StupidAE](https://huggingface.co/Muinez/StupidAE)

---

## Key Numbers

- Total params: **13,243,539**
- Compression: **d8 (8×8 patching)**
- Latent channels: **16 (c16)**
- Training: **30k steps**, batch size **256**, **~3** RTX 5090-hours
- Optimizer: **Muon + SnooC**, LR = `1e-3`
- Trained **without KL loss** (just mse)

---

## Performance (compared to SDXL VAE)

Stats for 1024×1024:

| Component | SDXL VAE | StupidAE |
|----------|----------|-----------|
| Encoder FLOPs | 4.34 TFLOPs | **124.18 GFLOPs** |
| Decoder FLOPs | 9.93 TFLOPs | **318.52 GFLOPs** |
| Encoder Params | 34.16M | **~3.8M** |
| Decoder Params | 49.49M | **~9.7M** |

The model is **tens of times faster and lighter**, making it usable directly inside training loops.

---

## Architecture Overview

### ❌ No Attention  
It is simply unnecessary for this design and only slows things down.

### 🟦 Encoder  
- Splits the image into **8×8 patches**  
- Each patch is encoded **independently**  
- Uses **only 1×1 convolutions**  
- Extremely fast

The encoder can handle any aspect ratio, but if you want to mix different ARs inside the same batch, the 1×1 conv version becomes inconvenient.
The Linear encoder version solves this completely — mixed batches work out of the box, although I haven’t released it yet — I can upload it if needed.

### 🟥 Decoder  
- Uses standard 3×3 convolutions (but 1×1 also works with surprisingly few artifacts)  
- Uses a **PixNeRF-style head** instead of stacked upsampling blocks

---

## Limitations

- Reconstruction is not perfect — small details may appear slightly blurred.  
- Current MSE loss: 0.0020.  
- This can likely be improved by increasing model size.

---

## Notes on 32× Compression

If you want **32× spatial compression**, do **not** use naive 32× patching — quality drops heavily.

A better approach:

1. First stage: patch-8 → 16/32 channels  
2. Second stage: patch-4 → ~256 channels  

This trains much better and works well for text-to-image training too.  
I’ve tested it, and the results are significantly more stable than naive approaches.

If you want to keep FLOPs low, you could try using patch-16 from the start, but I’m not sure yet how stable the training would be.

I’m currently working on a **d32c64** model with reconstruction quality better than Hunyuan VAE, but I’m limited by compute resources.

---

## Support the Project

I’m renting an **RTX 5090** and running all experiments on it.  
I’m currently looking for work and would love to join a team doing text-to-image or video model research.

If you want to support development:

- TRC20: 👉 TPssa5ung2MgqbaVr1aeBQEpHC3xfmm1CL  
- BTC: bc1qfv6pyq5dvs0tths682nhfdnmdwnjvm2av80ej4
- Boosty: https://boosty.to/muinez  

---

## How to use

Here's a minimal example:

```python
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import v2
from IPython.display import display
import requests
from stae import StupidAE

vae = StupidAE().cuda().half()
vae.load_state_dict(
    torch.load(hf_hub_download(repo_id="Muinez/StupidAE", filename="smol_f8c16.pt"))
)

t = v2.Compose([
    v2.Resize((1024, 1024)),
    v2.ToTensor(),
    v2.Normalize([0.5], [0.5])
])

image = Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG", stream=True).raw).convert("RGB")

with torch.inference_mode():
    image = t(image).unsqueeze(0).cuda().half()
    
    latents = vae.encode(image)
    image_decoded = vae.decode(latents)
    
    image = v2.ToPILImage()(torch.clamp(image_decoded * 0.5 + 0.5, 0, 1).squeeze(0))
    display(image)
```


## Coming Soon

- Linear-encoder variant  
- d32c64 model  
- Tutorial: training text-to-image **without bucketing** (supports mixed aspect ratios)

---

## Cite

```bibtex
@misc{StupidAE,
    title        = {StupidAE: A Tiny Patch-Based Autoencoder for Fast Image Compression},
    author       = {Muinez},
    year         = {2025},
    howpublished = \url{https://github.com/Muinez/StupidAE},
}
```

