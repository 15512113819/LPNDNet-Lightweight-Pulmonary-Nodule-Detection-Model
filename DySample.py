
class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4):
        super().__init__()
        self.scale = scale
        self.groups = groups

        assert style == 'lp'
        assert in_channels >= groups and in_channels % groups == 0

        out_channels = 2 * groups * scale ** 2
        self.offset = nn.Conv2d(in_channels, out_channels, 1)

        nn.init.normal_(self.offset.weight, mean=0, std=0.001)
        nn.init.constant_(self.offset.bias, 0)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1
        ) / self.scale

        return torch.stack(
            torch.meshgrid(h, h, indexing='ij')
        ).transpose(1, 2).repeat(
            1, self.groups, 1
        ).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape

        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H)
        coords_w = torch.arange(W)

        coords = torch.stack(
            torch.meshgrid(coords_w, coords_h, indexing='ij')
        ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)

        coords = coords + 0.5

        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device
        ).view(1, 2, 1, 1, 1)

        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(
            coords.view(B, -1, H, W), self.scale
        ).view(
            B, 2, -1, self.scale * H, self.scale * W
        ).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode='border'
        ).reshape(B, -1, self.scale * H, self.scale * W)

    def forward(self, x):
        offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)
