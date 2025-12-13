import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Function

# --- Pure Python Replacement for ailut_transform ---

def ailut_transform(img, luts, vertices):
    """
    Pure PyTorch implementation of Adaptive Interval 3D LUT transform.

    Args:
        img: Input image (B, 3, H, W), values in [0, 1].
        luts: LUT tensor (B, 3, D, D, D).
        vertices: Adaptive intervals (B, 3, D).

    Returns:
        output: Transformed image (B, 3, H, W).
    """
    B, C, H, W = img.shape
    D = vertices.shape[-1]

    # 1. Map Image Values to Normalized Coordinates using Vertices
    # Vertices are monotonic increasing [0, 1].
    # We map p -> coord in [-1, 1].

    # We process each channel.
    norm_img_list = []

    img_flat = img.view(B, C, -1) # (B, 3, N)

    for c in range(C):
        # We handle batch properly
        # For efficiency, loop over batch (inference usually B=1)
        # Or broadcast if B is large? B=1 is typical here.

        batch_coords = []
        for b in range(B):
            v_c = vertices[b, c, :] # (D)
            p_c = img_flat[b, c, :] # (N)

            # Find lower bound
            # searchsorted: v[i-1] <= p < v[i]
            # output index i in [0, D]
            idx = torch.searchsorted(v_c, p_c.contiguous(), right=True)
            idx = torch.clamp(idx, 1, D - 1)

            v_lo = v_c[idx - 1]
            v_hi = v_c[idx]

            # Fraction
            denom = v_hi - v_lo + 1e-8
            frac = (p_c - v_lo) / denom

            # Coordinate in [0, D-1]
            coord = (idx - 1).float() + frac

            # Normalize to [-1, 1]
            # 0 -> -1, D-1 -> 1
            # coord_norm = (coord / (D-1)) * 2 - 1
            coord_norm = (coord / (D - 1)) * 2.0 - 1.0

            batch_coords.append(coord_norm)

        norm_img_list.append(torch.stack(batch_coords, dim=0)) # (B, N)

    # Stack channels: (B, 3, N)
    norm_img = torch.stack(norm_img_list, dim=1)

    # 2. Prepare Grid for grid_sample
    # grid_sample input: (N, C, D_in, H_in, W_in) -> LUT (B, 3, D, D, D)
    # grid: (N, D_out, H_out, W_out, 3)
    # We map 2D image pixels to 3D LUT coordinates.
    # Effectively flattening the image to a list of points.
    # grid size: (B, 1, 1, N_pixels, 3)

    # Coordinate order for grid_sample is (x, y, z).
    # LUT dimensions are (R, G, B) based on meshgrid analysis.
    # (Dim0=R, Dim1=G, Dim2=B).
    # grid_sample (x, y, z) maps to (Dim2, Dim1, Dim0) -> (B, G, R).
    # So we need to stack (B_coords, G_coords, R_coords).

    # norm_img is (B, 3, N) -> (R, G, B) order.
    # grid should be (B, 1, 1, N, 3) with (B, G, R) order.

    grid = torch.stack([norm_img[:, 2, :], norm_img[:, 1, :], norm_img[:, 0, :]], dim=-1) # (B, N, 3)
    grid = grid.view(B, 1, 1, -1, 3) # (B, 1, 1, N, 3)

    # 3. Sample
    # luts must be (B, 3, D, D, D)
    # Note: LUTGenerator output is (B, 3, D, D, D).
    # grid_sample expects this.

    sampled = F.grid_sample(luts, grid, align_corners=True, mode='bilinear', padding_mode='border')
    # Output: (B, C, 1, 1, N)

    sampled = sampled.view(B, C, H, W)

    return sampled


# --- Components copied and cleaned from ailutmodel.py ---

class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.2)
        ]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)

class TPAMIBackbone(nn.Sequential):
    def __init__(self, pretrained=False, input_resolution=256, extra_pooling=False):
        body = [
            BasicBlock(3, 16, stride=2, norm=True),
            BasicBlock(16, 32, stride=2, norm=True),
            BasicBlock(32, 64, stride=2, norm=True),
            BasicBlock(64, 128, stride=2, norm=True),
            BasicBlock(128, 128, stride=2),
            nn.Dropout(p=0.5),
        ]
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = 128 * (4 if extra_pooling else 64)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)

class Res18Backbone(nn.Module):
    def __init__(self, pretrained=True, input_resolution=224, extra_pooling=False):
        super().__init__()
        # Note: Removing pretrained=True requirement or loading it from url if strictly needed.
        # But for inference with custom weights, we usually just define architecture.
        # However, torchvision resnet18(pretrained=True) downloads from internet.
        # Since we load our own weights into AiLUT (which wraps this), we might not need ImageNet weights.
        # But AiLUT loads 'AiLUT-FiveK-sRGB.pth' into the whole model.
        # So we just need the architecture.
        # To be safe regarding 'pretrained=True' logic in init, we keep it but it might warn.
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Identity()
        self.net = net
        self.input_resolution = input_resolution
        self.out_channels = 512

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return self.net(imgs).view(imgs.shape[0], -1)

class LUTGenerator(nn.Module):
    def __init__(self, n_colors, n_vertices, n_feats, n_ranks) -> None:
        super().__init__()
        self.weights_generator = nn.Linear(n_feats, n_ranks)
        self.basis_luts_bank = nn.Linear(n_ranks, n_colors * (n_vertices ** n_colors), bias=False)
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.n_feats = n_feats
        self.n_ranks = n_ranks
        self.init_weights()

    def init_weights(self):
        nn.init.ones_(self.weights_generator.bias)
        identity_lut = torch.stack([
            torch.stack(
                torch.meshgrid(*[torch.arange(self.n_vertices) for _ in range(self.n_colors)], indexing='ij'),
                dim=0).div(self.n_vertices - 1).flip(0),
            *[torch.zeros(
                self.n_colors, *((self.n_vertices,) * self.n_colors)) for _ in range(self.n_ranks - 1)]
            ], dim=0).view(self.n_ranks, -1)
        self.basis_luts_bank.weight.data.copy_(identity_lut.t())

    def forward(self, x, weights_delta = None):
        if weights_delta is not None:
            updated_params = torch.mul(self.weights_generator.weight, 1 + weights_delta)
            weights = F.linear(x, updated_params, self.weights_generator.bias)
        else:
            weights = F.linear(x, self.weights_generator.weight, self.weights_generator.bias)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

class AdaInt(nn.Module):
    def __init__(self, n_colors, n_vertices, n_feats, adaint_share=False) -> None:
        super().__init__()
        repeat_factor = n_colors if not adaint_share else 1
        self.intervals_generator = nn.Linear(n_feats, (n_vertices - 1) * repeat_factor)
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.adaint_share = adaint_share
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.intervals_generator.weight)
        nn.init.ones_(self.intervals_generator.bias)

    def forward(self, x, weights_delta = None):
        if weights_delta is not None:
            updated_params = torch.mul(self.intervals_generator.weight, 1 + weights_delta)
            intervals = F.linear(x, updated_params, self.intervals_generator.bias)
        else:
            intervals = F.linear(x, self.intervals_generator.weight, self.intervals_generator.bias)
        intervals = intervals.view(x.shape[0], -1, self.n_vertices - 1)
        if self.adaint_share:
            intervals = intervals.repeat_interleave(self.n_colors, dim=1)
        intervals = intervals.softmax(-1)
        vertices = F.pad(intervals.cumsum(-1), (1, 0), 'constant', 0)
        return vertices

class AiLUT(nn.Module):
    def __init__(self, n_ranks=3, n_colors=3, n_vertices=33, backbone='tpami', pretrained=False):
        super().__init__()
        self.n_ranks = n_ranks
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.en_adaint = True
        self.en_adaint_share = False
        self.backbone_name = backbone.lower()

        # mapping f
        if self.backbone_name == 'tpami':
            self.backbone = TPAMIBackbone(pretrained, extra_pooling=True)
        elif self.backbone_name == 'res18':
            self.backbone = Res18Backbone(pretrained, extra_pooling=True)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # mapping h
        self.lut_generator = LUTGenerator(
            n_colors, n_vertices, self.backbone.out_channels, n_ranks)

        # mapping g
        self.adaint = AdaInt(
            n_colors, n_vertices, self.backbone.out_channels, self.en_adaint_share)

        self.uniform_vertices = torch.arange(n_vertices).div(n_vertices - 1).repeat(n_colors, 1)

    def forward(self, lq, weights_deltas = None):
        if weights_deltas is not None:
            lut_weights_delta, adaint_weights_delta = weights_deltas
        else:
            lut_weights_delta = None
            adaint_weights_delta = None

        codes = self.backbone(lq)
        weights, luts = self.lut_generator(codes, lut_weights_delta)

        if self.en_adaint:
            vertices = self.adaint(codes, adaint_weights_delta)
        else:
            vertices = self.uniform_vertices.unsqueeze(0).to(lq.device)

        outs = ailut_transform(lq, luts, vertices)
        outs = torch.clamp(outs, 0, 1)
        return outs, weights, vertices


# --- Adaptation Module (Text Adapter) ---

class AdaptationModule(nn.Module):
    def __init__(self, n_ranks=3, n_vertices=33, n_colors=3, feature_dim=1024, n_feats=512) -> None:
        super().__init__()
        self.lut_weight_dim = (n_ranks, n_feats)
        self.adaint_weight_dim = ((n_vertices - 1) * n_colors, n_feats)

        self.block1 = nn.Linear(feature_dim, feature_dim)
        self.activation = nn.LeakyReLU()
        self.lut_delta_generator = nn.Linear(feature_dim, self.lut_weight_dim[0] * self.lut_weight_dim[1])
        self.adaint_delta_generator = nn.Linear(feature_dim, self.adaint_weight_dim[0] * self.adaint_weight_dim[1])
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.block1.weight, 0.0, 0.01)
        nn.init.zeros_(self.block1.bias)
        nn.init.uniform_(self.lut_delta_generator.weight, 0.0, 0.01)
        nn.init.uniform_(self.adaint_delta_generator.weight, 0.0, 0.01)
        nn.init.zeros_(self.lut_delta_generator.bias)
        nn.init.zeros_(self.adaint_delta_generator.bias)

    def forward(self, x, intensity=1):
        x = x.to(torch.float32)
        x = self.block1(x)
        x = self.activation(x)
        lut_weights_delta = self.lut_delta_generator(x)
        adaint_weights_delta = self.adaint_delta_generator(x)
        lut_weights_delta = lut_weights_delta.view(self.lut_weight_dim[0], self.lut_weight_dim[1])
        adaint_weights_delta = adaint_weights_delta.view(self.adaint_weight_dim[0], self.adaint_weight_dim[1])
        return (intensity * lut_weights_delta, intensity * adaint_weights_delta)
