import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => Instance Norm => Leaky ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(negative_slope=0.02, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
 

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, palette_embed, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Separate key and value projections
        self.linear_q = nn.Linear(palette_embed, embed_dim)  # Query projection for palette embedding
        self.linear_k = nn.Linear(embed_dim, embed_dim)  # Key projection for feature map
        self.linear_v = nn.Linear(embed_dim, embed_dim)  # Value projection for feature map

    def forward(self, x, palette_embedding):
        # Reshape and project for key and value
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(2, 0, 1)  # Shape: (h * w, b, c)
        k = self.linear_k(x_flat)  # Key projection
        v = self.linear_v(x_flat)  # Value projection
        # Expand palette_embedding and project for query
        b_, c_, h_, w_ = palette_embedding.shape
        palette_embedding = palette_embedding.view(b_, c_, h_ * w_).permute(2, 0, 1)
        q = self.linear_q(palette_embedding)

        # Apply multi-head attention with separate keys and values
        attn_output, _ = self.multihead_attn(q, k, v)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)  # Reshape back to (b, c, h, w)
        
        # Concatenate attention output with the original feature map
        return x + attn_output  # Concatenate along the channel dimension


def adjust_target_palettes(target_palettes_emb, h, w):
    # Starting with the initial target_palettes shape
    target_palettes = target_palettes_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
    return target_palettes


class RecoloringDecoder(nn.Module):
    def __init__(self, palette_embedding_dim=64, num_heads=1):
        super().__init__()
        self.palette_embedding_dim = palette_embedding_dim
        self.palette_fc = nn.Linear(4 * 24 * 3, palette_embedding_dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  

        # Cross-attention layers for palette conditioning at each decoding stage
        self.cross_attn_4 = CrossAttention(256, palette_embedding_dim, num_heads)
        self.cross_attn_3 = CrossAttention(128, palette_embedding_dim, num_heads)
        self.cross_attn_2 = CrossAttention(64, palette_embedding_dim, num_heads)
        self.cross_attn_1 = CrossAttention(64, palette_embedding_dim, num_heads)

        # DoubleConv layers for each decoding stage
        self.dconv_up_4 = DoubleConv(512, 256)
        self.dconv_up_3 = DoubleConv(512, 128)
        self.dconv_up_2 = DoubleConv(256, 64)
        self.dconv_up_1 = DoubleConv(128, 64)
        
        # Final convolutional layer
        self.conv_last = nn.Conv2d(64 + 1, 3, kernel_size=3, padding=1)

    def forward(self, c1, c2, c3, c4, target_palettes, illu):
        bz, _, _, _ = c1.shape
        # Flatten and project target_palettes to create a conditioning embedding
        target_palettes_flat = target_palettes.reshape(bz, -1)  # Shape: (bz, 4 * 16 * 4)
        palette_embedding = self.palette_fc(target_palettes_flat)  # Shape: (bz, palette_embedding_dim)

        # Decoder with cross-attention conditioning
        x = self.dconv_up_4(c1)
        palette_embedding_repeated = adjust_target_palettes(palette_embedding, x.size(-2), x.size(-1))
        x = self.cross_attn_4(x, palette_embedding_repeated)  # Apply cross-attention with palette embedding
        x = self.up(x)

        x = torch.cat([x, c2], dim=1)
        x = self.dconv_up_3(x)
        palette_embedding_repeated = adjust_target_palettes(palette_embedding, x.size(-2), x.size(-1))
        x = self.cross_attn_3(x, palette_embedding_repeated)  # Cross-attention at the next stage
        x = self.up(x)

        x = torch.cat([x, c3], dim=1)
        x = self.dconv_up_2(x)
        palette_embedding_repeated = adjust_target_palettes(palette_embedding, x.size(-2), x.size(-1))
        x = self.cross_attn_2(x, palette_embedding_repeated)  # Cross-attention at the next stage
        x = self.up(x)

        x = torch.cat([x, c4], dim=1)
        x = self.dconv_up_1(x)
        palette_embedding_repeated = adjust_target_palettes(palette_embedding, x.size(-2), x.size(-1))
        x = self.cross_attn_1(x, palette_embedding_repeated)  # Cross-attention at the final stage
        x = self.up(x)

        # Concatenate with illumination information
        illu = illu.view(illu.size(0), 1, illu.size(1), illu.size(2))
        x = torch.cat((x, illu), dim=1)
        x = self.conv_last(x)
        x = torch.tanh(x)
        return x
        
