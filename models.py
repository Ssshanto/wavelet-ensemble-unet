# Hyperparameters list
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

from transform2d import DWTForward, DWTInverse

__all__ = ['UNet', 'Wavelet_UNet']

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers = 4, n_filters = 64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, n_filters)
        self.downs = nn.ModuleList([Down(n_filters * 2**i, n_filters * 2**(i+1)) for i in range(n_layers)])
        self.ups = nn.ModuleList([Up(n_filters * 2**(n_layers-i), n_filters * 2**(n_layers-i-1), bilinear) for i in range(n_layers)])
        self.outc = OutConv(n_filters, n_classes)

    def forward(self, x):
        x_skip_connections = []
        x = self.inc(x)
        x_skip_connections.append(x)
        
        for down in self.downs:
            x = down(x)
            x_skip_connections.append(x)
        
        for i, up in enumerate(self.ups):
            x = up(x, x_skip_connections[-(i+2)])
        
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        for down in self.downs:
            down = torch.utils.checkpoint(down)
        for up in self.ups:
            up = torch.utils.checkpoint(up)
        self.outc = torch.utils.checkpoint(self.outc)

# Wavelet_UNet

class WaveletDownsampling(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super(WaveletDownsampling, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.dwt_forward = DWTForward(wave=self.wavelet, J=self.level)

    def forward(self, x):
        coeffs = self.dwt_forward(x)
        return coeffs[0]

class WaveletUpsampling(nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletUpsampling, self).__init__()
        self.wavelet = wavelet
        self.dwt_inverse = DWTInverse(wave=self.wavelet)
    
    def forward(self, cA):
        coeffs = (cA, [None, None, None])
        x = self.dwt_inverse(coeffs)
        return x

class Wavelet_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, wavelet):
        super().__init__()
        self.wavelet_conv = nn.Sequential(
            WaveletDownsampling(wavelet=wavelet, level=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.wavelet_conv(x)

class Wavelet_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, wavelet):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = WaveletUpsampling(wavelet=wavelet)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Wavelet_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(Wavelet_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Wavelet_Down(64, 128, wavelet))
        self.down2 = (Wavelet_Down(128, 256, wavelet))
        self.down3 = (Wavelet_Down(256, 256, wavelet))
        self.up1 = (Wavelet_Up(512, 128, wavelet))
        self.up2 = (Wavelet_Up(256, 64, wavelet))
        self.up3 = (Wavelet_Up(128, 64, wavelet))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)