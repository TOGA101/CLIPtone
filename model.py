import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Pure Python Replacement for ailut_transform ---

def ailut_transform(img, luts, vertices):
    """
    Pure PyTorch implementation of Adaptive Interval 3D LUT transform.
    """
    B, C, H, W = img.shape
    D = vertices.shape[-1]

    # 1. Map Image Values to Normalized Coordinates using Vertices
    norm_img_list = []
    img_flat = img.view(B, C, -1) # (B, 3, N)

    for c in range(C):
        batch_coords = []
        for b in range(B):
            v_c = vertices[b, c, :] # (D)
            p_c = img_flat[b, c, :] # (N)

            idx = torch.searchsorted(v_c, p_c.contiguous(), right=True)
            idx = torch.clamp(idx, 1, D - 1)

            v_lo = v_c[idx - 1]
            v_hi = v_c[idx]

            denom = v_hi - v_lo + 1e-8
            frac = (p_c - v_lo) / denom

            coord = (idx - 1).float() + frac
            coord_norm = (coord / (D - 1)) * 2.0 - 1.0

            batch_coords.append(coord_norm)

        norm_img_list.append(torch.stack(batch_coords, dim=0))

    norm_img = torch.stack(norm_img_list, dim=1)

    # 2. Prepare Grid for grid_sample
    grid = norm_img.permute(0, 2, 1) # (B, N, 3) -> (R, G, B)
    grid = grid.view(B, 1, 1, -1, 3) # (B, 1, 1, N, 3)

    # 3. Sample
    sampled = F.grid_sample(luts, grid, align_corners=True, mode='bilinear', padding_mode='border')
    sampled = sampled.view(B, C, H, W)

    return sampled


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
    def __init__(self, input_resolution=256):
        body = [
            BasicBlock(3, 16, stride=2, norm=True),
            BasicBlock(16, 32, stride=2, norm=True),
            BasicBlock(32, 64, stride=2, norm=True),
            BasicBlock(64, 128, stride=2, norm=True),
            BasicBlock(128, 128, stride=2),
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(2)
        ]
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = 128 * 4

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2,
            mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)

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

    def forward(self, x, weights_delta):
        updated_params = torch.mul(self.weights_generator.weight, 1 + weights_delta)
        weights = F.linear(x, updated_params, self.weights_generator.bias)
        luts = self.basis_luts_bank(weights)
        luts = luts.view(x.shape[0], -1, *((self.n_vertices,) * self.n_colors))
        return weights, luts

class AdaInt(nn.Module):
    def __init__(self, n_colors, n_vertices, n_feats) -> None:
        super().__init__()
        repeat_factor = n_colors
        self.intervals_generator = nn.Linear(n_feats, (n_vertices - 1) * repeat_factor)
        self.n_colors = n_colors
        self.n_vertices = n_vertices
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.intervals_generator.weight)
        nn.init.ones_(self.intervals_generator.bias)

    def forward(self, x, weights_delta):
        updated_params = torch.mul(self.intervals_generator.weight, 1 + weights_delta)
        intervals = F.linear(x, updated_params, self.intervals_generator.bias)
        intervals = intervals.view(x.shape[0], -1, self.n_vertices - 1)
        intervals = intervals.softmax(-1)
        vertices = F.pad(intervals.cumsum(-1), (1, 0), 'constant', 0)
        return vertices

class AiLUT(nn.Module):
    def __init__(self, n_ranks=3, n_vertices=33):
        super().__init__()
        self.n_ranks = n_ranks
        self.n_colors = 3
        self.n_vertices = n_vertices

        # Hardcoded TPAMI
        self.backbone = TPAMIBackbone()

        # mapping h
        self.lut_generator = LUTGenerator(
            self.n_colors, n_vertices, self.backbone.out_channels, n_ranks)

        # mapping g
        self.adaint = AdaInt(
            self.n_colors, n_vertices, self.backbone.out_channels)

    def forward(self, lq, weights_deltas):
        lut_weights_delta, adaint_weights_delta = weights_deltas

        codes = self.backbone(lq)
        weights, luts = self.lut_generator(codes, lut_weights_delta)
        vertices = self.adaint(codes, adaint_weights_delta)

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
