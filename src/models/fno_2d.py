"""2D Fourier Neural Operator (FNO) for Spatio-Temporal Physical PDEs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Fourier Layer: Computes spectral convolution via FFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to retain along spatial dim 1
        self.modes2 = modes2  # Number of Fourier modes to retain along spatial dim 2

        self.scale = 1.0 / (in_channels * out_channels)
        # Learnable complex weights for lowest modes
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        batchsize = x.shape[0]

        # 1. Compute 2D Real FFT
        x_ft = torch.fft.rfft2(x)

        # 2. Multiply relevant Fourier modes by complex weights
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Upper frequency corner
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1,
        )
        # Lower frequency corner
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2,
        )

        # 3. Return to spatial domain via Inverse 2D Real FFT
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    """Full 2D Fourier Neural Operator architecture.

    Maps input function a(x) -> u(x) across continuous spatial domains.
    Input channels: [T_t, Y_t, elevation, grad_x, grad_y, wind_x, wind_y]
    Output channels: [T_{t+1}, Y_{t+1}]
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Lift input features to higher-dimensional latent channel space
        self.fc0 = nn.Conv2d(in_channels, self.width, kernel_size=1)

        # Spectral convolution blocks
        self.convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(num_layers)
        ])
        # Local bypass spatial convolutions (W in FNO formulation)
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=1)
            for _ in range(num_layers)
        ])

        # Projection back to target physical field dimensions
        self.fc1 = nn.Conv2d(self.width, 64, kernel_size=1)
        self.fc2 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature projection
        x = self.fc0(x)

        # Fourier operator blocks
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)

        # Final projection to physical fields
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
