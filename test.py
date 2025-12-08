import torch
import time
import argparse
import os
from torchvision.utils import save_image
from ailutmodel import AiLUT
from adaptation import AdaptationModule
from criteria import CLIPLoss
from PIL import Image
from torchvision import transforms

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
    parser.add_argument('--save_path', type=str, default='./test')
    parser.add_argument('--clip_model', type=str, default='RN50')
    parser.add_argument('--intensity', type=float, default=1)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--input_image', type=str, default='input.jpg')
    
    
    device = 'cpu'
    print(f'Using device: {device}')

    args = parser.parse_args()

    # Load image
    if not os.path.exists(args.input_image):
        print(f"Generating synthetic image for testing: {args.input_image}")
        img = Image.new('RGB', (480, 480), color = 'red')
        img.save(args.input_image)

    lq = Image.open(args.input_image).convert('RGB')
    transform = transforms.ToTensor()
    lq = transform(lq).unsqueeze(0).to(device)

    
    CLIPloss = CLIPLoss(device, clip_model = args.clip_model)
    
    model = AiLUT(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.backbone_checkpoint_dir, args.backbone_checkpoint_name), map_location=device)['state_dict'])
    model.requires_grad_(False)
    model.eval()
    adaptation_module = AdaptationModule(args, CLIPloss.model.text_projection.shape[1], model.backbone.out_channels).to(device)
    adaptation_module.load_state_dict(torch.load(os.path.join(args.adaptor_checkpoint_dir, args.adaptor_checkpoint_name), map_location=device))
    adaptation_module.requires_grad_(False)
    adaptation_module.eval()
    
    
    save_path = os.path.join(args.save_path, args.prompt)
    target_text = args.prompt + ' photo.'
    text_direction_features = CLIPloss.compute_text_direction('Normal photo.', target_text)

    # Inference
    weights_deltas = adaptation_module(text_direction_features, args.intensity)
    out, _, _ = model(lq, weights_deltas = weights_deltas)
    file_name = os.path.splitext(os.path.basename(args.input_image))[0]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_image(out, os.path.join(save_path, f'{file_name}.png'))
    print(f"Inference complete. Result saved to {os.path.join(save_path, f'{file_name}.png')}")
        
if __name__ == '__main__':
    main()
