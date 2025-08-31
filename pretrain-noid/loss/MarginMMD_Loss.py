import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


class MarginMMD_Loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, P=8, K=4, margin=None):
        super(MarginMMD_Loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.P = P
        self.K = K
        self.margin = margin
        if self.margin:
            print(f'Using Margin : {self.margin}')
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) + 1e-9 for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        if torch.sum(torch.isnan(sum(kernel_val))):
            ## We encountered a Nan in Kernel
            print(f'Bandwidth List : {bandwidth_list}')
            print(f'L2 Distance : {L2_distance}')
            ## Check for Nan in L2 distance
            print(f'L2 Nan : {torch.sum(torch.isnan(L2_distance))}')
            for bandwidth_temp in bandwidth_list:
                print(f'Temp: {bandwidth_temp}')
                print(f'BW Nan : {torch.sum(torch.isnan(L2_distance / bandwidth_temp))}')
        return sum(kernel_val), L2_distance

    def forward(self, source, target, labels1=None, labels2=None):
        ## Source  - [P*K, 2048], Target - [P*K, 2048]
        ## Devide them in "P" groups of "K" images
        rgb_features_list, ir_features_list = list(torch.split(source, [self.K] * self.P, dim=0)), list(
            torch.split(target, [self.K] * self.P, dim=0))
        total_loss = torch.tensor([0.], requires_grad=True).to(torch.device('cuda'))
        if labels1 is not None and labels2 is not None:
            rgb_labels, ir_labels = torch.split(labels1, [self.K] * self.P, dim=0), torch.split(labels2,
                                                                                                [self.K] * self.P,
                                                                                                dim=0)
            print(f'RGB Labels : {rgb_labels}')
            print(f'IR Labels : {ir_labels}')

        xx_batch, yy_batch, xy_batch, yx_batch = 0, 0, 0, 0

        for rgb_feat, ir_feat in zip(rgb_features_list, ir_features_list):
            source, target = rgb_feat, ir_feat  ## 4, 2048 ; 4*2048 -> 4*2048
            ## (rgb, ir, mid) -> rgb - mid + ir- mid ->
            batch_size = int(source.size()[0])
            kernels, l2dist = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul,
                                                   kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]

            xx_batch += torch.mean(XX)
            yy_batch += torch.mean(YY)
            xy_batch += torch.mean(XY)
            yx_batch += torch.mean(YX)

            if self.margin:
                loss = torch.mean(XX + YY - XY - YX)
                if loss - self.margin > 0:
                    total_loss += loss
                else:
                    total_loss += torch.clamp(loss - self.margin, min=0)

            else:
                total_loss += torch.mean(XX + YY - XY - YX)

        total_loss /= self.P
        return total_loss, torch.max(l2dist), [xx_batch / self.P, yy_batch / self.P, xy_batch / self.P,
                                               yx_batch / self.P]