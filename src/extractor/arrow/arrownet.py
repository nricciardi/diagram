import torch
import torch.nn as nn


class ArrowNet(nn.Module):

    @staticmethod
    def double_conv(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def __init__(self, in_ch=1, base_ch=32, n_classes=3):
        super().__init__()

        # Downsampling
        self.encoder1 = ArrowNet.double_conv(in_ch, base_ch)
        self.encoder2 = ArrowNet.double_conv(base_ch, base_ch * 2)
        self.pooling = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ArrowNet.double_conv(base_ch*2, base_ch*4)

        # Upsampling
        self.upsampling2 = ArrowNet.up_block(base_ch * 4, base_ch * 2)
        self.decoder2 = ArrowNet.double_conv(base_ch * 4, base_ch * 2)
        self.upsampling1 = ArrowNet.up_block(base_ch * 2, base_ch)
        self.decoder1 = ArrowNet.double_conv(base_ch * 2, base_ch)

        self.output_conv = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)  # (B, base, 256,256)
        e2 = self.encoder2(self.pooling(e1))  # (B, base*2,128,128)

        b = self.bottleneck(self.pooling(e2))  # (B, base*4,64,64)

        d2 = self.upsampling2(b)  # (B, base*2,128,128)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upsampling1(d2)  # (B, base,256,256)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        return self.output_conv(d1)  # (B,3,256,256)
