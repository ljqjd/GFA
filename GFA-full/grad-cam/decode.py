import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
def reshape_and_average(input_tensor):
    reshaped_tensor = input_tensor.view(21, 10, 768)
    averaged_tensor = reshaped_tensor.mean(dim=2)
    expanded_tensor = torch.zeros(256*124)
    
    top_left_point = averaged_tensor[0, 0].repeat(14).unsqueeze(-1).expand(-1, 14)
    top_right_point = averaged_tensor[0, -1].repeat(14).unsqueeze(-1).expand(-1, 14)
    bottom_left_point = averaged_tensor[-1, 0].repeat(14).unsqueeze(-1).expand(-1, 14)
    bottom_right_point = averaged_tensor[-1, -1].repeat(14).unsqueeze(-1).expand(-1, 14)
    
    
    # 将 14x14 的块放置在 [21*16, 10*16] 的张量中的四个角
    expanded_tensor[:14, :14] = top_left_point.flatten()
    expanded_tensor[:14, -14:] = top_right_point.flatten()
    expanded_tensor[-14:, :14] = bottom_left_point.flatten()
    expanded_tensor[-14:, -14:] = bottom_right_point.flatten()
    
    for i in range(2, 8):  # 8个是因为去掉了两个角点，10-2=8
        point = averaged_tensor[0, i].repeat(14).unsqueeze(-1).expand(-1, 12)
        expanded_tensor[:14, 16*i-14:16*i] = point.flatten()
        point = averaged_tensor[-1, i].repeat(14).unsqueeze(-1).expand(-1, 12)
        expanded_tensor[-14:, 16*i-14:16*i] = point.flatten()
    
    for i in range(2, 19, 16):  # 19个是因为去掉了两个角点，21-2=19，步长为16是因为每个块宽14，间隔2
        point = averaged_tensor[:, 0].repeat(12, 1).transpose(0, 1)
        expanded_tensor[i-2:i+14, :14] = point[:14, :]
        point = averaged_tensor[:, -1].repeat(12, 1).transpose(0, 1)
        expanded_tensor[i-2:i+14, -14:] = point[:14, :]

    for i in range(1, 20):
        for j in range(1, 9):
            point = averaged_tensor[i, j].repeat(12).unsqueeze(-1).expand(-1, 12)
            expanded_tensor[16*i-12:16*i+4, 16*j-12:16*j+4] = point.flatten()
    
    
    plt.figure(figsize=(10, 6))  # 设置图的大小
    plt.imshow(output_tensor.cpu().numpy(), cmap='hot')  # 使用热力图颜色映射
    plt.colorbar()  # 显示颜色条
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    return expanded_tensor

