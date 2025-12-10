import torch
import torch.nn as nn
import os
import argparse
from torch.export import export, Dim
from executorch.exir import to_edge, EdgeCompileConfig

# Import models
from ailutmodel import AiLUT
from adaptation import AdaptationModule
from criteria import CLIPLoss

# Import pure PyTorch implementation and monkey patch
from ailut_transform_pt import ailut_transform as ailut_transform_pt
import ailutmodel
ailutmodel.ailut_transform = ailut_transform_pt

class WrapperModel(nn.Module):
    def __init__(self, ailut_model, adaptation_module, intensity):
        super().__init__()
        self.ailut_model = ailut_model
        self.adaptation_module = adaptation_module
        self.intensity = intensity

    def forward(self, lq, lq_small, text_direction_features):
        # lq: full resolution image for transform
        # lq_small: 256x256 image for backbone

        codes = self.ailut_model.backbone(lq_small)
        weights_deltas = self.adaptation_module(text_direction_features, self.intensity)

        # mapping h
        # weights_deltas is (lut_weights_delta, adaint_weights_delta)
        lut_weights_delta, adaint_weights_delta = weights_deltas
        weights, luts = self.ailut_model.lut_generator(codes, lut_weights_delta)

        # mapping g
        if self.ailut_model.en_adaint:
            vertices = self.ailut_model.adaint(codes, adaint_weights_delta)
        else:
            vertices = self.ailut_model.uniform_vertices.unsqueeze(0) # .to(device) handled by runtime?

        outs = ailutmodel.ailut_transform(lq, luts, vertices)
        outs = torch.clamp(outs, 0, 1)

        return outs

def export_model(args):
    device = 'cpu'

    # Load Models
    print("Loading models...")
    # Dummy CLIP to get dimensions
    text_feat_dim = 1024

    model = AiLUT(args).to(device)
    model.load_state_dict(torch.load(os.path.join(args.backbone_checkpoint_dir, args.backbone_checkpoint_name), map_location=device)['state_dict'])
    model.eval()

    adaptation_module = AdaptationModule(args, text_feat_dim, model.backbone.out_channels).to(device)
    adaptation_module.load_state_dict(torch.load(os.path.join(args.adaptor_checkpoint_dir, args.adaptor_checkpoint_name), map_location=device))
    adaptation_module.eval()

    # Patch TPAMIBackbone forward to skip internal interpolate
    from ailutmodel import TPAMIBackbone
    def backbone_forward_no_interp(self, imgs):
        return super(TPAMIBackbone, self).forward(imgs).view(imgs.shape[0], -1)

    import types
    model.backbone.forward = types.MethodType(backbone_forward_no_interp, model.backbone)

    wrapper = WrapperModel(model, adaptation_module, args.intensity)
    wrapper.eval()

    # Example inputs
    example_lq = torch.randn(1, 3, 512, 512)
    example_lq_small = torch.randn(1, 3, 256, 256)
    example_text_features = torch.randn(1, 1024)

    print("Tracing...")

    height = Dim("height", min=128, max=4096)
    width = Dim("width", min=128, max=4096)

    dynamic_shapes = {
        "lq": {0: 1, 2: height, 3: width},
        "lq_small": {0: 1, 2: 256, 3: 256}, # Fixed size for backbone
        "text_direction_features": {0: 1}
    }

    exported_program = export(
        wrapper,
        (example_lq, example_lq_small, example_text_features),
        dynamic_shapes=dynamic_shapes
    )

    print("To Edge...")
    # Add searchsorted to exception list just in case, though we removed it.
    edge_prog = to_edge(
        exported_program,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False
        )
    )

    print("To ExecuTorch...")
    executorch_prog = edge_prog.to_executorch()

    output_path = "ailut_model.pte"
    with open(output_path, "wb") as f:
        f.write(executorch_prog.buffer)

    print(f"Exported model to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    # Same args as test.py
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
    parser.add_argument('--intensity', type=float, default=1)

    args = parser.parse_args()

    export_model(args)

if __name__ == '__main__':
    main()
