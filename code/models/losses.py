import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from skimage import color
from typing import Literal

class VGGPerceptual(nn.Module):
    def __init__(self, device:Literal['cpu', 'cuda', 'mps']):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg.to(device)

    def forward(self, pred_rgb, target_rgb):
        f1 = self.vgg(pred_rgb)
        f2 = self.vgg(target_rgb)
        return nn.functional.l1_loss(f1, f2)

def lab_to_rgb_tensor(L, ab):
    B = L.size(0)
    out = []
    for i in range(B):
        Li = L[i,0].detach().cpu().numpy() * 100.0
        abi = ab[i].detach().cpu().numpy().transpose(1,2,0) * 128.0
        lab = np.concatenate([Li[:,:,None], abi], axis=2)
        rgb = color.lab2rgb(lab)
        out.append(torch.from_numpy(rgb.transpose(2,0,1)).float())
    return torch.stack(out, dim=0)

def validate(model, loader, device, l1_loss, perc_loss):
    model.eval()
    total_loss = 0
    total_l1 = 0
    total_perc = 0
    num_batches = len(loader)
    with torch.no_grad():
        for L, ab in loader:
            L, ab = L.to(device), ab.to(device)
            pred_ab = model(L)
            loss_l1 = l1_loss(pred_ab, ab)
            target_rgb = lab_to_rgb_tensor(L, ab).to(device)
            pred_rgb = lab_to_rgb_tensor(L, pred_ab).to(device)
            loss_perc = perc_loss(pred_rgb, target_rgb)
            loss = loss_l1 + 0.1 * loss_perc
            total_loss += loss.item()
            total_l1 += loss_l1.item()
            total_perc += loss_perc.item()
    return total_loss / num_batches, total_l1 / num_batches, total_perc / num_batches