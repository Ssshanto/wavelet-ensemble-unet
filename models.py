# Hyperparameters list
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

from transform2d import DWTForward, DWTInverse

__all__ = ['UNet', 'Wavelet_UNet', 'MR_UNet']

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
            DoubleConv(in_channels, out_channels)
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
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
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


class Wavelet_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(Wavelet_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Wavelet_Down(64, 128, wavelet))
        self.down2 = (Wavelet_Down(128, 256, wavelet))
        self.down3 = (Wavelet_Down(256, 512, wavelet))
        self.down4 = (Wavelet_Down(512, 512, wavelet))
        self.up1 = (Wavelet_Up(1024, 256, wavelet))
        self.up2 = (Wavelet_Up(512, 128, wavelet))
        self.up3 = (Wavelet_Up(256, 64, wavelet))
        self.up4 = (Wavelet_Up(128, 64, wavelet))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
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

# Wavelet_UNet with all 4 channels in downsampling

class Wavelet_Down_All(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, wavelet):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            WaveletDownsamplingAll(wavelet=wavelet, level=1),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Wavelet_UNet_All(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(Wavelet_UNet_All, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Wavelet_Down_All(4 * 64, 128, wavelet))
        self.down2 = (Wavelet_Down_All(4 * 128, 256, wavelet))
        self.down3 = (Wavelet_Down_All(4 * 256, 512, wavelet))
        self.down4 = (Wavelet_Down_All(4 * 512, 512, wavelet))
        self.up1 = (Wavelet_Up(1024, 256, wavelet))
        self.up2 = (Wavelet_Up(512, 128, wavelet))
        self.up3 = (Wavelet_Up(256, 64, wavelet))
        self.up4 = (Wavelet_Up(128, 64, wavelet))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
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
    
# MRUnet

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

class MR_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(MR_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.wavelet_downsample = WaveletDownsampling(wavelet=wavelet, level=1)
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.concat1 = MRConcat(n_channels, 128)
        self.down2 = (Down(128, 256))
        self.concat2 = MRConcat(n_channels, 256)
        self.down3 = (Down(256, 512))
        self.concat3 = MRConcat(n_channels, 512)
        self.down4 = (Down(512, 1024))
        self.concat4 = MRConcat(n_channels, 1024)
        
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        downsampled2 = self.wavelet_downsample(x)
        downsampled3 = self.wavelet_downsample(downsampled2)
        downsampled4 = self.wavelet_downsample(downsampled3)
        downsampled5 = self.wavelet_downsample(downsampled4)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.concat1(downsampled2, x2)
        x3 = self.down2(x2)
        x3 = self.concat2(downsampled3, x3)
        x4 = self.down3(x3)
        x4 = self.concat3(downsampled4, x4)
        x5 = self.down4(x4)
        x5 = self.concat4(downsampled5, x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

# MR-UNet with addition instead of concatenation

class MRConcatAdd(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MRConcatAdd, self).__init__()
        self.conv1 = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x = x1 + x2
        return x

class MR_UNet_Add(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(MR_UNet_Add, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.wavelet_downsample = WaveletDownsampling(wavelet=wavelet, level=1)
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.concat1 = MRConcatAdd(n_channels, 128)
        self.down2 = (Down(128, 256))
        self.concat2 = MRConcatAdd(n_channels, 256)
        self.down3 = (Down(256, 512))
        self.concat3 = MRConcatAdd(n_channels, 512)
        self.down4 = (Down(512, 1024))
        self.concat4 = MRConcatAdd(n_channels, 1024)
        
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        downsampled2 = self.wavelet_downsample(x)
        downsampled3 = self.wavelet_downsample(downsampled2)
        downsampled4 = self.wavelet_downsample(downsampled3)
        downsampled5 = self.wavelet_downsample(downsampled4)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.concat1(downsampled2, x2)
        x3 = self.down2(x2)
        x3 = self.concat2(downsampled3, x3)
        x4 = self.down3(x3)
        x4 = self.concat3(downsampled4, x4)
        x5 = self.down4(x4)
        x5 = self.concat4(downsampled5, x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
# MR-UNet with HH Information

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

class MR_UNet_HH(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(MR_UNet_HH, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.wavelet_downsample = WaveletDownsamplingHH(wavelet=wavelet, level=1)
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.concat1 = MRConcatAdd(1, 128)
        self.down2 = (Down(128, 256))
        self.concat2 = MRConcatAdd(1, 256)
        self.down3 = (Down(256, 512))
        self.concat3 = MRConcatAdd(1, 512)
        self.down4 = (Down(512, 1024))
        self.concat4 = MRConcatAdd(1, 1024)
        
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        downsampled2 = self.wavelet_downsample(x)
        downsampled3 = self.wavelet_downsample(downsampled2)
        downsampled4 = self.wavelet_downsample(downsampled3)
        downsampled5 = self.wavelet_downsample(downsampled4)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.concat1(downsampled2, x2)
        x3 = self.down2(x2)
        x3 = self.concat2(downsampled3, x3)
        x4 = self.down3(x3)
        x4 = self.concat3(downsampled4, x4)
        x5 = self.down4(x4)
        x5 = self.concat4(downsampled5, x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

# Wavelet UNet with all 4 channel information (LL, LH, HL, HH)

class WaveletDownsamplingAll(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super(WaveletDownsamplingAll, self).__init__()
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

class MR_UNet_All(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(MR_UNet_All, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.wavelet_downsample = WaveletDownsamplingAll(wavelet=wavelet, level=1)
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(4 * 64, 128))
        self.concat1 = MRConcatAdd(4, 128)
        self.down2 = (Down(4 * 128, 256))
        self.concat2 = MRConcatAdd(4, 256)
        self.down3 = (Down(4 * 256, 512))
        self.concat3 = MRConcatAdd(4, 512)
        self.down4 = (Down(4 * 512, 1024))
        self.concat4 = MRConcatAdd(4, 1024)
        
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        downsampled2 = self.wavelet_downsample(x)
        # take only the LL portion of each downsample, use 0:1 to preserve dimension ordering
        downsampled3 = self.wavelet_downsample(downsampled2[:, 0:1])
        downsampled4 = self.wavelet_downsample(downsampled3[:, 0:1])
        downsampled5 = self.wavelet_downsample(downsampled4[:, 0:1])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.concat1(downsampled2, x2)
        x3 = self.down2(x2)
        x3 = self.concat2(downsampled3, x3)
        x4 = self.down3(x3)
        x4 = self.concat3(downsampled4, x4)
        x5 = self.down4(x4)
        x5 = self.concat4(downsampled5, x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits