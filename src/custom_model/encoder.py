import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => Instance Norm => Leaky ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SelfAttention(nn.Module):
    """Applies self-attention within a feature map."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Query, Key, Value projections for self-attention mechanism
        self.linear_q = nn.Linear(embed_dim, embed_dim)  # Query projection
        self.linear_k = nn.Linear(embed_dim, embed_dim)  # Key projection
        self.linear_v = nn.Linear(embed_dim, embed_dim)  # Value projection

    def forward(self, x):
        b, c, h, w = x.shape
        # Reshape and project feature map
        x_flat = x.view(b, c, h * w).permute(2, 0, 1)  # Shape: (h * w, batch, channels)
        
        q = self.linear_q(x_flat)  # Query projection
        k = self.linear_k(x_flat)  # Key projection
        v = self.linear_v(x_flat)  # Value projection

        # Multi-head attention
        attn_output, _ = self.multihead_attn(q, k, v)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)

        # Concatenate attention output with the original feature map
        return torch.cat([x, attn_output], dim=1)  # Concatenate along the channel dimension

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(inplace=True)
        self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetBasicBlock(ResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # Define the shortcut path if dimensions do not match
        self.shortcut = nn.Sequential()
        if downsampling != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=downsampling, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # Either the original input or downsampled version
        out = self.blocks(x)
        out += identity  # Add the skip connection (F(x) + x)
        return F.relu(out)

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels),
            *[block(out_channels * block.expansion, out_channels) for _ in range(n - 1)]
        )

    def forward(self, x):
        return self.blocks(x)

class FeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, num_heads=4):
        super().__init__()

        # DoubleConv layers for each encoding stage
        self.dconv_down_1 = DoubleConv(in_channels, 64)
        self.dconv_down_2 = DoubleConv(128, 128)
        self.dconv_down_3 = DoubleConv(256, 256)
        self.dconv_down_4 = DoubleConv(256, 512)

        # ResNet layers for each encoding stage
        self.res1 = ResNetLayer(64, 128)
        self.res2 = ResNetLayer(128, 256)
        self.res3 = ResNetLayer(256, 512)

        # Self-attention layers for each encoding stage
        self.self_attn_1 = SelfAttention(128, num_heads)
        self.self_attn_2 = SelfAttention(256, num_heads)
        self.self_attn_3 = SelfAttention(512, num_heads)

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoding stage 1
        x = self.dconv_down_1(x)
        x = self.res1(x)
        x = self.self_attn_1(x)
        c1 = self.pool(x)  # Save for use in the decoder

        # Encoding stage 2
        x = self.dconv_down_2(c1)
        x = self.res2(x)
        x = self.self_attn_2(x)
        c2 = self.pool(x)

        # Encoding stage 3
        x = self.dconv_down_3(c2)
        x = self.res3(x)
        x = self.self_attn_3(x)
        c3 = self.pool(x)

        # Encoding stage 4 (without further downsampling)
        c4 = self.dconv_down_4(c3)

        return c1, c2, c3, c4


