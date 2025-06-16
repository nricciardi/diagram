import torch
import torch.nn as nn


class ArrowNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input: (B, 1, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Upsampling
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 3, kernel_size=1)  # Output logits (B, 3, H, W)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        y = self.decoder(x)

        return torch.softmax(y, dim=1)
