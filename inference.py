import argparse

import clip
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from model import AiLUT, AdaptationModule


def load_image(path):
    # Load image and handle EXIF orientation
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    return img

def main():
    parser = argparse.ArgumentParser(description="CPU Inference for CLIPtone")
    parser.add_argument('--input', type=str, default='test.jpg', help='Path to input image')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save output image')
    parser.add_argument('--prompt', type=str, default='Normal', help='Target text description')
    parser.add_argument('--intensity', type=float, default=1.0, help='Intensity of the effect')
    parser.add_argument('--base_checkpoint', type=str, default='checkpoint/base_network/AiLUT-FiveK-sRGB.pth')
    parser.add_argument('--adaptor_checkpoint', type=str, default='checkpoint/text_adaptor/RN50/pretrained.pth')
    parser.add_argument('--clip_checkpoint', type=str, default='checkpoint/clip/RN50.pt', help='Path to CLIP model checkpoint')

    args = parser.parse_args()

    device = "cpu"
    print(f"Running inference on {device}...")

    # 1. Load CLIP Model (for text direction)
    print("Loading CLIP model...")
    # Using RN50 as per original repo
    clip_model, clip_preprocess = clip.load(args.clip_checkpoint, device=device)

    def get_text_features(text):
        tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            features = clip_model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    # Compute Text Direction
    source_text = "Normal photo."
    target_text = args.prompt + ' photo.'
    print(f"Computing direction from '{source_text}' to '{target_text}'...")

    source_features = get_text_features(source_text)
    target_features = get_text_features(target_text)
    text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
    text_direction /= text_direction.norm(dim=-1, keepdim=True) # (1, 1024)

    # 2. Load AiLUT Model
    print("Loading AiLUT model...")
    # Params match original args defaults: n_ranks=3, n_vertices=33, backbone='tpami'
    model = AiLUT(n_ranks=3, n_vertices=33)

    # Load weights
    checkpoint = torch.load(args.base_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device)
    model.eval()

    # 3. Load Adaptation Module
    print("Loading Adaptation Module...")
    # Feature dim for RN50 is 1024. Backbone out channels (n_feats) is 512 for TPAMI.
    adaptor = AdaptationModule(n_ranks=3, n_vertices=33, n_colors=3, feature_dim=1024, n_feats=512)

    adaptor.load_state_dict(torch.load(args.adaptor_checkpoint, map_location=device))

    adaptor.to(device)
    adaptor.eval()

    # 4. Inference
    print(f"Processing {args.input}...")
    img_pil = load_image(args.input)

    # Resize logic? Original uses SingleImageDataset which just loads.
    # But TPAMIBackbone interpolates internally to input_resolution (256) for the codes,
    # but the final transform is applied to the full resolution image (or however lq is passed).
    # In test.py: `out, _, _ = model(lq, ...)`
    # lq comes from DataLoader. `SingleImageDataset` transforms?
    # Original dataset uses `TF.to_tensor(img)`. No resizing in dataset for validation usually.
    # AiLUT code: `codes = self.backbone(lq)`. Backbone interpolates to 256 internally.
    # `outs = ailut_transform(lq, luts, vertices)`. lq is full res.
    # So we just convert to tensor.

    img_tensor = ToTensor()(img_pil).unsqueeze(0).to(device) # (1, 3, H, W)

    with torch.no_grad():
        weights_deltas = adaptor(text_direction, intensity=args.intensity)
        out, _, _ = model(img_tensor, weights_deltas=weights_deltas)

    # Save
    save_image(out, args.output)
    print(f"Saved output to {args.output}")

if __name__ == "__main__":
    main()
