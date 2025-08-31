import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)



class Normalize(nn.Module):
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim

    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-4)
        return out



class DistributionPredictor(nn.Module):
    def __init__(self):
        super(DistributionPredictor, self).__init__()
        
        dim_mlp = 768
        out_dim = 192
        self.linear_mean = nn.Linear(dim_mlp, out_dim)
        self.linear_var = nn.Linear(dim_mlp, out_dim)

        self.conv_var = nn.Conv2d(3, 3, kernel_size=(8, 8), bias=False)

        self.n_samples = torch.Size(np.array([2, ]))
        self.pooling_layer = GeneralizedMeanPoolingP(3)

        self.l2norm_mean, self.l2norm_var, self.l2norm_sample = Normalize(2, 1), Normalize(2, 1), Normalize(2, 2)

        self.bottleneck = nn.BatchNorm2d(out_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.relu = nn.ReLU()



    def forward(self, x):
        
        out_mean = self.pooling_layer(x)  
        out_mean = out_mean.view(out_mean.size(0), -1)  
        out_mean = self.linear_mean(x) 
        out_mean = self.l2norm_mean(out_mean)

        #out_var = x.unsqueeze(0)
        out_var = x.view(32,3,16,16)
        out_var = self.conv_var(out_var)  # conv layer
        out_var = self.pooling_layer(out_var)  # pooling
        out_var += 1e-4
        out_var = out_var.view(out_var.size(0), -1)  
        out_var = self.linear_var(x) 
        out_var = self.relu(out_var) + 1e-4
        out_var = self.l2norm_var(out_var)
        

        return out_mean, out_var  

