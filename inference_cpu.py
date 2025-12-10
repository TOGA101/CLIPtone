import torch
import argparse
import os
from torchvision.transforms import ToTensor
from PIL import Image, ImageOps
from ailutmodel import AiLUT
from adaptation import AdaptationModule
from criteria import CLIPLoss

# Import the pure PyTorch implementation
from ailut_transform_pt import ailut_transform as ailut_transform_pt

# Monkey patch the ailut_transform in ailutmodel to use the PyTorch version
import ailutmodel
ailutmodel.ailut_transform = ailut_transform_pt

def load_image(image_path, device):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    transform = ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, os.path.basename(image_path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='tpami')
    parser.add_argument('--backbone_checkpoint_dir', type=str, default='./checkpoint/base_network')
    parser.add_argument('--backbone_checkpoint_name', type=str, default='AiLUT-FiveK-sRGB.pth')
    parser.add_argument('--n_ranks', type=int, default=3)
    parser.add_argument('--n_vertices', type=int, default=33)
    parser.add_argument('--en_adaint', type=bool, default=True)
    parser.add_argument('--en_adaint_share', type=bool, default=False)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--n_colors', type=int, default=3)
    parser.add_argument('--adaptor_checkpoint_dir', type=str, default='./checkpoint/RN50')
    parser.add_argument('--adaptor_checkpoint_name', type=str, default='pretrained.pth')
    parser.add_argument('--save_path', type=str, default='./result')
    parser.add_argument('--clip_model', type=str, default='RN50')
    parser.add_argument('--intensity', type=float, default=1)
    parser.add_argument('--prompt', type=str, default='Dark') # Default prompt

    args = parser.parse_args()

    device = 'cpu' # Force CPU

    print(f"Running on {device}")

    # Load CLIP for text direction
    print("Loading CLIP...")
    CLIPloss = CLIPLoss(device, clip_model = args.clip_model)

    # Load Models
    print("Loading AiLUT...")
    model = AiLUT(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.backbone_checkpoint_dir, args.backbone_checkpoint_name), map_location=device)['state_dict'])
    model.requires_grad_(False)
    model.eval()

    print("Loading AdaptationModule...")
    text_proj_dim = CLIPloss.model.text_projection.shape[1]
    adaptation_module = AdaptationModule(args, text_proj_dim, model.backbone.out_channels).to(device)
    adaptation_module.load_state_dict(torch.load(os.path.join(args.adaptor_checkpoint_dir, args.adaptor_checkpoint_name), map_location=device))
    adaptation_module.requires_grad_(False)
    adaptation_module.eval()

    # Input
    img_path = 'test.jpg'
    lq, file_name = load_image(img_path, device)
    print(f"Loaded image {file_name} with shape {lq.shape}")

    target_text = args.prompt + ' photo.'
    print(f"Computing text direction for '{target_text}'...")
    text_direction_features = CLIPloss.compute_text_direction('Normal photo.', target_text)

    print("Inference...")
    start_time = os.times().elapsed
    with torch.no_grad():
        weights_deltas = adaptation_module(text_direction_features, args.intensity)
        out, _, _ = model(lq, weights_deltas = weights_deltas)
    end_time = os.times().elapsed

    print(f"Inference done in {end_time - start_time:.4f}s")

    save_path = os.path.join(args.save_path, args.prompt)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    out_path = os.path.join(save_path, f'{os.path.splitext(file_name)[0]}_cpu.png')
    from torchvision.utils import save_image
    save_image(out, out_path)
    print(f"Saved result to {out_path}")

if __name__ == '__main__':
    main()
