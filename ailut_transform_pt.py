import torch
import torch.nn.functional as F

def ailut_transform(img, lut, vertices):
    """
    Pure PyTorch implementation of AiLUT transform compatible with ExecuTorch (no searchsorted).
    Uses unrolled loop to avoid memory explosion.

    Args:
        img (torch.Tensor): Input image of shape (b, 3, h, w). Range [0, 1].
        lut (torch.Tensor): Output values of the 3D LUT, shape (b, 3, d, d, d).
        vertices (torch.Tensor): Sampling coordinates along each dimension of
            the 3D LUT, shape (b, 3, d). Assumed to be sorted.

    Returns:
        torch.Tensor: Transformed image of shape (b, 3, h, w).
    """
    b, c, h, w = img.shape
    d = vertices.shape[-1]

    # Flatten image for processing: (b, 3, n) where n = h * w
    img_flat = img.view(b, c, -1)

    # Find indices such that vertices[idx] <= val < vertices[idx+1]
    # We unroll the search to avoid searchsorted and memory explosion.

    # count = number of vertices <= val
    # Initialize count to 0
    # Use IntTensor for counting
    count = torch.zeros((b, c, h*w), dtype=torch.int32, device=img.device)

    # Iterate over vertices. d is usually 33.
    for i in range(d):
        # v_i: (b, 3) -> (b, 3, 1)
        v_i = vertices[:, :, i].unsqueeze(2)

        # Compare: (b, 3, n)
        # We assume vertices are sorted ascending.
        # if val >= v[i], we increment count.
        mask = (img_flat >= v_i)
        count = count + mask.to(torch.int32)

    # idx = count - 1
    idx = count - 1
    idx = torch.clamp(idx, min=0, max=d - 2)

    # Convert to int64 for gather
    idx = idx.to(torch.int64)

    # idx has shape (b, 3, n)
    # Unpack indices
    rid = idx[:, 0, :] # (b, n)
    gid = idx[:, 1, :] # (b, n)
    bid = idx[:, 2, :] # (b, n)

    # Helper to gather values
    def gather_vertices(channel_idx, indices):
        v = vertices[:, channel_idx, :]
        return torch.gather(v, 1, indices)

    r0 = gather_vertices(0, rid)
    r1 = gather_vertices(0, rid + 1)
    g0 = gather_vertices(1, gid)
    g1 = gather_vertices(1, gid + 1)
    b0 = gather_vertices(2, bid)
    b1 = gather_vertices(2, bid + 1)

    # Compute weights
    r = img_flat[:, 0, :]
    g = img_flat[:, 1, :]
    b_val = img_flat[:, 2, :]

    eps = 1e-6
    rd = (r - r0) / (r1 - r0 + eps)
    gd = (g - g0) / (g1 - g0 + eps)
    bd = (b_val - b0) / (b1 - b0 + eps)

    w000 = (1 - rd) * (1 - gd) * (1 - bd)
    w100 = (    rd) * (1 - gd) * (1 - bd)
    w010 = (1 - rd) * (    gd) * (1 - bd)
    w110 = (    rd) * (    gd) * (1 - bd)
    w001 = (1 - rd) * (1 - gd) * (    bd)
    w101 = (    rd) * (1 - gd) * (    bd)
    w011 = (1 - rd) * (    gd) * (    bd)
    w111 = (    rd) * (    gd) * (    bd)

    stride_d = d
    stride_d2 = d * d

    base_id = rid + gid * stride_d + bid * stride_d2

    id000 = base_id
    id100 = base_id + 1
    id010 = base_id + stride_d
    id110 = base_id + 1 + stride_d
    id001 = base_id + stride_d2
    id101 = base_id + 1 + stride_d2
    id011 = base_id + stride_d + stride_d2
    id111 = base_id + 1 + stride_d + stride_d2

    lut_flat = lut.view(b, c, -1)

    def gather_lut(indices):
        indices_exp = indices.unsqueeze(1).expand(-1, c, -1)
        return torch.gather(lut_flat, 2, indices_exp)

    l000 = gather_lut(id000)
    l100 = gather_lut(id100)
    l010 = gather_lut(id010)
    l110 = gather_lut(id110)
    l001 = gather_lut(id001)
    l101 = gather_lut(id101)
    l011 = gather_lut(id011)
    l111 = gather_lut(id111)

    out = w000.unsqueeze(1) * l000 + \
          w100.unsqueeze(1) * l100 + \
          w010.unsqueeze(1) * l010 + \
          w110.unsqueeze(1) * l110 + \
          w001.unsqueeze(1) * l001 + \
          w101.unsqueeze(1) * l101 + \
          w011.unsqueeze(1) * l011 + \
          w111.unsqueeze(1) * l111

    out = out.view(b, c, h, w)

    return out
