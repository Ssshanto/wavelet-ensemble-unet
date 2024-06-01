# Hyperparameters list
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

from transform2d import DWTForward, DWTInverse

# Absolute Necessities

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

    def __init__(self, in_channels, out_channels, bilinear=False):
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
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class WaveletDownsampling(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super(WaveletDownsampling, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.dwt_forward = DWTForward(wave=self.wavelet, J=self.level)

    def forward(self, x):
        coeffs = self.dwt_forward(x)
        ll = coeffs[0]
        ll = (ll - ll.min()) / (ll.max() - ll.min())
        return ll

class WaveletUpsampling(nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletUpsampling, self).__init__()
        self.wavelet = wavelet
        self.dwt_inverse = DWTInverse(wave=self.wavelet)
    
    def forward(self, cA):
        coeffs = (cA, [None])
        x = self.dwt_inverse(coeffs)
        return x

class Wavelet_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, wavelet):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            WaveletDownsampling(wavelet=wavelet, level=1),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Wavelet_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, wavelet):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = WaveletUpsampling(wavelet=wavelet)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


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
    
class WaveletDownsamplingAll1C(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super(WaveletDownsamplingAll1C, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.dwt_forward = DWTForward(wave=self.wavelet, J=self.level)
        self.conv = DoubleConv(4, 1)

    def forward(self, x):
        coeffs = self.dwt_forward(x)
        # for each channel of x, extract all 4 dwt components ll, hl, lh, hh

        ll = coeffs[0]
        lh = coeffs[1][0][:, :, 0]
        hl = coeffs[1][0][:, :, 1]
        hh = coeffs[1][0][:, :, 2]
        # normalize all 4 channels
        ll = (ll - ll.min()) / (ll.max() - ll.min())
        lh = (lh - lh.min()) / (lh.max() - lh.min())
        hl = (hl - hl.min()) / (hl.max() - hl.min())
        hh = (hh - hh.min()) / (hh.max() - hh.min())

        # stack the channels together
        stacked = torch.stack([ll, lh, hl, hh], dim=2)
        
        batch_size, channels, _, height, width = stacked.shape
        
        stacked_reshaped = stacked.view(batch_size * channels, 4, height, width)
        convolved_stacked = self.conv(stacked_reshaped)
        f = convolved_stacked.view(batch_size, channels, height, width)
        return f

class WaveletDownsamplingAll4C(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super(WaveletDownsamplingAll4C, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.dwt_forward = DWTForward(wave=self.wavelet, J=self.level)

    def forward(self, x):
        coeffs = self.dwt_forward(x)
        # extract all 4 channels, ll, hl, lh, hh
        ll = coeffs[0]
        lh = coeffs[1][0][:, :, 0]
        hl = coeffs[1][0][:, :, 1]
        hh = coeffs[1][0][:, :, 2]
        # normalize all 4 channels
        ll = (ll - ll.min()) / (ll.max() - ll.min())
        lh = (lh - lh.min()) / (lh.max() - lh.min())
        hl = (hl - hl.min()) / (hl.max() - hl.min())
        hh = (hh - hh.min()) / (hh.max() - hh.min())
        # concatenate the 4 channels
        f = torch.cat([ll, lh, hl, hh], dim=1)
        return f

class Wavelet_Down_All(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, wavelet):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            WaveletDownsamplingAll1C(wavelet=wavelet, level=1),
            nn.BatchNorm2d(in_channels),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

# MR UNet Functions

class MRConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MRConcat, self).__init__()
        self.conv1 = DoubleConv(in_channels, out_channels)
        self.conv2 = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        return x
    
class MRConcatAdd(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MRConcatAdd, self).__init__()
        self.conv1 = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x = x1 + x2
        return x
    
class MR_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels=4, wavelet='db1'):
        super().__init__()
        self.downsample = nn.Sequential(
            WaveletDownsamplingAll4C(wavelet=wavelet, level=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

class WaveletDownsamplingHH(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super(WaveletDownsamplingHH, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.dwt_forward = DWTForward(wave=self.wavelet, J=self.level)

    def forward(self, x):
        coeffs = self.dwt_forward(x)
        hh = coeffs[1][0][:, :, 2]
        hh = (hh - hh.min()) / (hh.max() - hh.min())
        return hh
    
# Wavelet Fusion Functions

class LL_HH_Fusion(nn.Module):
    def __init__(self):
        super(LL_HH_Fusion, self).__init__()

    def forward(self, x1, x2):
        x = x1 + x2
        return x