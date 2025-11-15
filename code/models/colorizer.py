import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

#As encoder part of the model we use pretrained ResNet34
class EncoderResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet34(pretrained=pretrained)
        self.initial = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)  # (3, 64, 64) -> (64, 16, 16)
        self.layer1 = res.layer1  # (64, 16, 16) -> (64, 16, 16)
        self.layer2 = res.layer2  # (128, 16, 16) -> (128, 8, 8)
        self.layer3 = res.layer3  # (256, 8, 8) -> (256, 4, 4)
        self.layer4 = res.layer4  # (256, 4, 4) -> (512, 2, 2)

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4

#As decoder part of the model we use UNet which will be trained
class DecoderUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def up(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'), #XXX nn.ConvTranspose2d
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.up4 = up(512, 256) # (512, 2, 2) -> (256, 4, 4) | encoder.layer4
        self.up3 = up(256 + 256, 128) # (256, 4, 4) + (256, 4, 4) -> (128, 8, 8) | self.up4 + encoder.layer3
        self.up2 = up(128 + 128, 64) #(128, 8, 8) + (128, 8, 8) -> (64, 16, 16) | self.up3 + encoder.layer2 
        self.up1 = up(64 + 64, 64) #(64, 16, 16) + (64, 16, 16) -> (64, 32, 32) | self.up2 + encoder.layer1
        #XXX рассмотреть возможность добавить еще один up слой чтобы не interpolate в ColorizerNet классе
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), #(64, 32, 32) -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 2, 1)  # predict ab (2 channels)
        )

    def forward(self, feats):
        x0, x1, x2, x3, x4 = feats
        x = self.up4(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        out = self.final(x)
        out = torch.tanh(out)
        return out

class ColorizerNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc = EncoderResNet34(pretrained=pretrained)
        self.dec = DecoderUNet()

    def forward(self, L):
        L3 = L.repeat(1, 3, 1, 1)
        feats = self.enc(L3)
        ab = self.dec(feats)
        ab = nn.functional.interpolate(ab, size=L.shape[2:], mode="bilinear", align_corners=False)
        return ab
