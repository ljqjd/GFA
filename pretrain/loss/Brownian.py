import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import *
from PIL import Image
from torch import nn
       
    
class brownian_bridge_loss(nn.Module):
    def __init__(self):
        super(brownian_bridge_loss, self).__init__()
    
    def forward(self, Z_V, Z_T, Z_aV, t):
        # 计算条件均值
        norm = t * Z_V + (1 - t) * Z_T
        mu = (t * Z_V + (1 - t) * Z_T) / norm
        
        # 计算损失
        loss = torch.mean((Z_aV - mu) ** 2)
        return loss
    