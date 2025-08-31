import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch
import torch.nn as nn
import heapq


class Select(nn.Module):
    def __init__(self):
        super(Select, self).__init__()
 
    def cross_entropy(self, original, now, label):
        loss = abs(-torch.sum(label * torch.log(now), dim=-1) + torch.sum(label * torch.log(original), dim=-1))
        return loss

    def forward(self, input1, input2, score1, score2, label):
        n = input1.size(0)  #16
        results = []
        for i in range(n):
            result = self.cross_entropy(score1[i], score2[i], label[i])
            results.append(result)   

        smallest_8 = heapq.nsmallest(n//2, enumerate(results), key=lambda x: x[1].item())
        smallest_8_indices = [idx for idx, val in smallest_8]    
        inputss = input2[smallest_8_indices]
        labelss = label[smallest_8_indices]
        return inputss, labelss