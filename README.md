# CLIPtone CPU Inference

This is a simplified CPU-only implementation of [CLIPtone](https://github.com/hmin970922/CLIPtone) for single image inference.

## Prerequisites

- Python 3.12
- Internet access (for downloading models and CLIP)

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download pre-trained models:

   **Base Network (AiLUT):**
   Download `AiLUT-FiveK-sRGB.pth` and place it in `checkpoint/base_network/`.
   ```bash
   mkdir -p checkpoint/base_network/
   wget -O checkpoint/base_network/AiLUT-FiveK-sRGB.pth https://github.com/ImCharlesY/AdaInt/raw/main/pretrained/AiLUT-FiveK-sRGB.pth
   ```

   **Text Adapter:**
   Download `pretrained.pth` and place it in `checkpoint/text_adaptor/RN50/` (or any path, specify in arguments).
   You can use `gdown` (installed via requirements) to download from Google Drive.
   ```bash
   mkdir -p checkpoint/text_adaptor/RN50/
   gdown -O checkpoint/text_adaptor/RN50/pretrained.pth https://drive.google.com/uc?id=171NTXGgme8AmSJJyy1F4hEE3OnRBQuql
   ```

## Usage

Run the inference script on a single image.

```bash
python inference_cpu.py --input test.jpg --output output.png --prompt "A vibrant and bright"
```

### Arguments

- `--input`: Path to input image (default: `test.jpg`).
- `--output`: Path to save output image (default: `output.png`).
- `--prompt`: Target text description (default: `Normal photo.`).
- `--intensity`: Intensity of the effect (default: `1.0`).
- `--base_checkpoint`: Path to base AiLUT checkpoint.
- `--adaptor_checkpoint`: Path to text adapter checkpoint.

## Notes

- This implementation uses a pure PyTorch version of the 3D LUT transform, removing the need for CUDA extensions (`ailut_transform`).
- It runs entirely on CPU.
- Ensure you have the model weights downloaded before running.
