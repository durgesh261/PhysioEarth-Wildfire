"""Standard 2D U-Net Baseline Architecture (Unconstrained Spatial CNN)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineUNet(nn.Module):
    """Standard 2D U-Net for wildfire state prediction.

    Represents prevailing deep learning practice without spectral operators or physical constraints.
    """

    def __init__(self, in_channels: int = 7, out_channels: int = 2, base_features: int = 24):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))

        self.up1 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 4, base_features * 2)

        self.up2 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 2, base_features)

        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x_up1 = self.up1(x3)
        x_up1 = torch.cat([x2, x_up1], dim=1)
        x_up1 = self.conv_up1(x_up1)

        x_up2 = self.up2(x_up1)
        x_up2 = torch.cat([x1, x_up2], dim=1)
        x_up2 = self.conv_up2(x_up2)

        logits = self.outc(x_up2)
        return logits
