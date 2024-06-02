# Hyperparameters list
from os import name
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from helper import *

from transform2d import DWTForward, DWTInverse

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

# Wavelet_UNet with all 4 channels in downsampling

class Wavelet_UNet_All(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(Wavelet_UNet_All, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Wavelet_Down_All(64, 128, wavelet))
        self.down2 = (Wavelet_Down_All(128, 256, wavelet))
        self.down3 = (Wavelet_Down_All(256, 512, wavelet))
        self.down4 = (Wavelet_Down_All(512, 512, wavelet))
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

# MRUnet

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

class MR_UNet_All(nn.Module):
    def __init__(self, n_channels, n_classes, wavelet='db1'):
        super(MR_UNet_All, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.wavelet_downsample = MR_Down(wavelet=wavelet)
        
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.concat1 = MRConcat(4, 128)
        self.down2 = (Down(128, 256))
        self.concat2 = MRConcat(4, 256)
        self.down3 = (Down(256, 512))
        self.concat3 = MRConcat(4, 512)
        self.down4 = (Down(512, 1024))
        self.concat4 = MRConcat(4, 1024)
        
        self.up1 = (Up(1024, 512))
        self.up2 = (Up(512, 256))
        self.up3 = (Up(256, 128))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        downsampled2 = self.wavelet_downsample(x)
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
    
class Feature_Fusion_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_layers = 4, n_filters = 64, wavelet='db1'):
        super(Feature_Fusion_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_layers = n_layers

        self.upsample = WaveletUpsampling(wavelet=wavelet)
        self.feature_extract = WaveletChannels(wavelet=wavelet) 

        self.inc = DoubleConv(n_channels, n_filters)

        self.downs = nn.ModuleList([Down(n_filters * 2**i, n_filters * 2**(i+1)) for i in range(n_layers)])
        self.ups = nn.ModuleList([FeatureFusionUp(n_filters * 2**(n_layers-i), n_filters * 2**(n_layers-i-1)) for i in range(n_layers)])
        self.outc = OutConv(n_filters, n_classes)

    def forward(self, x):
        x = self.upsample(x)
        ll, lh, hl, hh = self.feature_extract(x)

        skip_connections = []
        ll = self.inc(ll)
        lh = self.inc(lh)
        hl = self.inc(hl)
        hh = self.inc(hh)

        skip_connections.append((ll, lh, hl, hh))
        x = ll

        for down in self.downs:
            x = down(x)
            lh = down(lh)
            hl = down(hl)
            hh = down(hh)
            skip_connections.append((x, lh, hl, hh))

        for i, up in enumerate(self.ups):
            ll, lh, hl, hh = skip_connections[-(i+2)]
            x = up(x, ll, lh, hl, hh)
        
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    model = Feature_Fusion_UNet(n_channels=1, n_classes=1).cuda()
    x = torch.randn(2, 1, 128, 128).cuda()
    y = model(x)