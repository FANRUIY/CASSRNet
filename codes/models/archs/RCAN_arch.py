import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# RCANBlock
# --------------------------
class RCANBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RCANBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# --------------------------
# CA_SA_Layer
# --------------------------
class CA_SA_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_SA_Layer, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_out = self.ca(x) * x
        sa_out = self.sa(x) * x
        return ca_out + sa_out

# --------------------------
# RCAB with NaN Check
# --------------------------
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, growth_rate=16, n_layers=4, res_scale=1):
        super(RCAB, self).__init__()
        self.n_layers = n_layers
        self.growth_rate = growth_rate
        self.res_scale = res_scale

        modules = []
        in_channels = n_feat
        for _ in range(n_layers):
            modules.append(nn.Sequential(
                RCANBlock(in_channels, growth_rate, kernel_size),
                nn.ReLU(inplace=True)
            ))
            in_channels += growth_rate

        self.layers = nn.ModuleList(modules)
        self.ca_sa = CA_SA_Layer(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, n_feat, 1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            input_cat = torch.cat(features, dim=1)
            out = layer(input_cat)
            features.append(out)
        out = torch.cat(features, dim=1)
        out = torch.clamp(out, min=-10, max=10)
        out = self.ca_sa(out)
        out = self.conv1x1(out)
        return x + out * self.res_scale

# --------------------------
# Residual Group
# --------------------------
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, n_resblocks, growth_rate=16, res_scale=1):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(n_feat, kernel_size, growth_rate, res_scale=res_scale) for _ in range(n_resblocks)
        ]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res + x

# --------------------------
# 修复尺寸不一致的 WindowAttentionBlock
# --------------------------
class WindowAttentionBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        orig_size = (H, W)

        need_pad = H % ws != 0 or W % ws != 0
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if need_pad:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = x.shape[-2:]

        x = x.view(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, ws * ws, C)

        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))

        x = x.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)

        if need_pad:
            x = x[:, :, :orig_size[0], :orig_size[1]]

        return x

# --------------------------
# RCAN Model
# --------------------------
class RCAN(nn.Module):
    def __init__(self, n_resgroups=5, n_resblocks=10, n_feats=64, kernel_size=3, scale=2, growth_rate=16, res_scale=1):
        super(RCAN, self).__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(3, n_feats, kernel_size, padding=kernel_size // 2)

        # Body
        modules_body = []
        for _ in range(n_resgroups):
            modules_body.append(ResidualGroup(
                n_feats, kernel_size, n_resblocks, growth_rate, res_scale=res_scale
            ))
            modules_body.append(WindowAttentionBlock(dim=n_feats, window_size=8))
        modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*modules_body)

        # Pyramid branch
        self.pyramid = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, stride=2, padding=kernel_size // 2),
            ResidualGroup(n_feats, kernel_size, max(1, n_resblocks // 2), growth_rate, res_scale=res_scale),
            nn.ConvTranspose2d(n_feats, n_feats, kernel_size=4, stride=2, padding=1)
        )

        # Tail
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, kernel_size, padding=kernel_size // 2),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, 3, kernel_size, padding=kernel_size // 2)
        )

    def forward(self, x):
        x_head = self.head(x)
        res = self.body(x_head)
        pyramid = self.pyramid(x_head)

        if pyramid.shape[2:] != res.shape[2:]:
            pyramid = F.interpolate(pyramid, size=res.shape[-2:], mode='bilinear', align_corners=False)

        res = res + pyramid
        res = res + x_head
        out = self.tail(res)
        return out
