import torch
import argparse
import os
from torchvision.transforms import ToTensor
from PIL import Image, ImageOps
from executorch.runtime import Runtime
from criteria import CLIPLoss

def load_image(image_path, device):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    transform = ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, os.path.basename(image_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='ailut_model.pte')
    parser.add_argument('--image_path', type=str, default='test.jpg')
    parser.add_argument('--prompt', type=str, default='Dark')
    parser.add_argument('--clip_model', type=str, default='RN50')
    parser.add_argument('--save_path', type=str, default='./result')

    args = parser.parse_args()

    # 1. Compute CLIP features (Standard PyTorch)
    print("Computing CLIP features...")
    device = "cpu"
    CLIPloss = CLIPLoss(device, clip_model = args.clip_model)
    target_text = args.prompt + ' photo.'
    text_direction_features = CLIPloss.compute_text_direction('Normal photo.', target_text)

    # 2. Load ExecuTorch Model
    print(f"Loading ExecuTorch model from {args.model_path}...")

    runtime = Runtime.get()
    program = runtime.load_program(args.model_path)

    # Load image
    lq, file_name = load_image(args.image_path, device)
    print(f"Image shape: {lq.shape}")

    # Resize to 256x256 for backbone input
    lq_small = torch.nn.functional.interpolate(lq, size=(256, 256), mode='bilinear', align_corners=False)

    # We need to construct the inputs.
    text_direction_features = text_direction_features.float()

    # Create method
    method_name = "forward"
    method = program.load_method(method_name)

    print("Running Inference...")

    # Inputs: lq, lq_small, text_direction_features
    outputs = method.execute([lq, lq_small, text_direction_features])

    # outputs is a list of tensors
    out = outputs[0]

    print("Inference done.")

    # Save result
    save_path = os.path.join(args.save_path, args.prompt)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    out_path = os.path.join(save_path, f'{os.path.splitext(file_name)[0]}_executorch.png')
    from torchvision.utils import save_image
    save_image(out, out_path)
    print(f"Saved result to {out_path}")

    # Compare with PyTorch output if it exists
    pt_path = os.path.join(save_path, f'{os.path.splitext(file_name)[0]}_cpu.png')
    if os.path.exists(pt_path):
        pt_out = ToTensor()(Image.open(pt_path))
        # Ensure sizes match
        if pt_out.shape != out.shape:
             pt_out = torch.nn.functional.interpolate(pt_out.unsqueeze(0), size=out.shape[2:], mode='bilinear').squeeze(0)

        # Calculate diff
        diff = (out - pt_out).abs().mean()
        print(f"Mean Absolute Difference with PyTorch output: {diff.item()}")

        if diff.item() < 1e-2:
            print("Verification PASSED: Outputs are very close.")
        else:
            print("Verification WARNING: Outputs differ.")

if __name__ == "__main__":
    main()
