import numpy as np
import torch
import torch.nn as nn

class GeneticAlgorithm(nn.Module):
    def __init__(self):
        super(GeneticAlgorithm, self).__init__()

    def arithmetic_crossover(self, parent1, parent2, crossover_rate):
        #crossover_rate = np.random.rand()
        summ = parent1.size(0)
        n = int(crossover_rate * summ)
        start_index = np.random.randint(0, summ - n)  # 交换开始的索引（0 基础）
        end_index = start_index + n  # 交换结束的索引（不包括该索引）

        # 交换 parent1 和 parent2 中的部分元素
        temp = parent1[start_index:end_index].clone()  # 创建一个临时变量来存储 parent1 的部分元素
        parent1[start_index:end_index] = parent2[start_index:end_index]  # 将 parent2 的部分元素复制到 parent1
        parent2[start_index:end_index] = temp  # 将临时变量中的元素复制到 parent2
        return parent1, parent2

    def mutation(self, population):
        with torch.no_grad():
            mutation_rate = 0.01
            average = population.mean()
            maxx = population.max()
            minn = population.min()

            for individual in range(population.size(0)):
                if torch.rand(1) < mutation_rate:
                # 使用torch.randn生成正态分布的随机数张量，形状与population[individual]相同
                    noise = torch.rand(1).cuda() * average
            
                # 将正态分布的随机数加到population[individual]上
                    population[individual] = population[individual] + noise
            
                # 使用torch.clamp确保变异后的值在[minn, maxx]范围内
                    population[individual] = torch.clamp(population[individual], min=minn, max=maxx)
    
        return population




    def forward(self, inputss, crossover_rate):
        n = inputss.size(0)
       # inputss = inputs.clone()
        for iii in range(n//2):
            inputss[iii], inputss[iii+n//2] = self.arithmetic_crossover(inputss[iii], inputss[iii+n//2], crossover_rate)
            self.mutation(inputss[iii])
            self.mutation(inputss[iii+n//2])
        return inputss





