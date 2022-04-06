r"""
    Copyright (C) 2022  Mark Locherer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# adapted from https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


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


def pad_img(in_tensor, out_tensor):
    r"""
    Pad input_tensor of shape N x C x H_i x W_i to shape of
    out tensor (N x C x Ht x Wt),
    for the first up convolution layer the target is derived from
    stage 4 (x4) w/ shape 64 x 64 and concatenated w/ 56 x 56 (x6).
    The tensor x4 is cropped to match the shape of x6.
    Args:
        in_tensor: Tensor to crop to match the shape of the target tensor
        out_tensor: Tensor to derive shape from.
    Returns:

    """
    # N, C, H, W (H = W)
    d_h = out_tensor.size()[2] - in_tensor.size()[2]
    d_w = out_tensor.size()[3] - in_tensor.size()[3]
    return F.pad(in_tensor, (d_w // 2, d_w - d_w // 2,
                             d_h // 2, d_h - d_h // 2))


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
        # here we pad in contrast to the original paper not the tnesor that is shared over layers, but the one that is
        # coming from below
        x1 = pad_img(x1, x2)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetSlim(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNetSlim, self).__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # decoder
        x3_d = self.up2(x4, x3)
        x2_d = self.up3(x3_d, x2)
        x1_d = self.up4(x2_d, x1)

        logits = self.outc(x1_d)

        return logits


if __name__ == "__main__":
    # random image w/ batchsize = 1, channels = 1, shape (h x w) 572 x 572
    image = torch.rand((1, 1, 572, 572))
    model = UNetSlim(1, 4)

    output = model(image)
    print(output.shape)

    # parameters
    unet_p = sum(p.numel() for p in model.parameters())
    unet_trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('total parameters: ', unet_p, 'trainable parameters: ', unet_trainable_p)
