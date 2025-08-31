import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms
import cv2

class GeneticAlgorithm(nn.Module):
    def __init__(self):
        super(GeneticAlgorithm, self).__init__()
        self.in_planes = 768
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def arithmetic_crossover(self, parent1, parent2):
        summ = parent1.size(0)  #210kuai
        child1 = attention_mechanism(parent1, parent2)
        child2 = attention_mechanism(parent2, parent1)
        return child1, child2

    def arithmetic_crossover2(self, parent1, parent2):
        summ = parent1.size(0)  #210kuai
        k = 40
        random_index = torch.randperm(summ)[:k]
        random_index2 = torch.randperm(summ)[:k]
        with torch.no_grad():
            for i in range(k):
                parent1[random_index[i],:] = parent2[random_index2[i],:]
        return parent1

    def mutation(self, population, gaussian_distribution):
        with torch.no_grad():
            random_points = gaussian_distribution.sample((1,))
            random_points = torch.squeeze(random_points)
            k = 20 
          #  k = random.randint(10, 30)
            random_index = torch.randperm(population.size(0))[:k]  #A
            population1 = population.clone()
            for i in range(k):
                    population[random_index[i]] = random_points[random_index[i]]
        return population


    def forward(self, inputss):
        n = inputss.size(0)  #64
        for iii in range(0, n//2, 4):
            if iii+3 < n//2 and iii+3+n//2 < n :
              #  if torch.rand(1) < 0.6:
                selected_elements = inputss[iii:iii+3]
                selected_elements1 = inputss[iii+n//2:iii+n//2+3]
                combined_elements = torch.cat([selected_elements, selected_elements1], dim=0)
                    
                mean = torch.mean(combined_elements, dim=0, keepdim=True)
                var = torch.var(combined_elements, dim=0, keepdim=True)
                std_dev = torch.sqrt(var + 1e-8)
                gaussian_distribution = torch.distributions.Normal(mean, std_dev)
              
                inputss[iii], inputss[iii+n//2] = self.arithmetic_crossover(inputss[iii], inputss[iii+n//2])
                inputss[iii+1+n//2], inputss[iii+1] = self.arithmetic_crossover(inputss[iii+1+n//2], inputss[iii+1])
                inputss[iii+2], inputss[iii+3] = self.arithmetic_crossover(inputss[iii+2], inputss[iii+3])
                inputss[iii+2+n//2], inputss[iii+3+n//2] = self.arithmetic_crossover(inputss[iii+2+n//2], inputss[iii+3+n//2])
                
         #       if torch.rand(1) < 0.6:
         #           numbers = [i for i in range(0, n) if i not in (iii,iii+1,iii+2,iii+3,iii+n//2,iii+n//2+1,iii+n//2+2,iii+n//2+3)]
         #           random_number = random.choice(numbers)
         #           inputss[iii+3] = self.arithmetic_crossover2(inputss[iii+3], inputss[random_number])
         #           inputss[iii+3+n//2] = self.arithmetic_crossover2(inputss[iii+3+n//2], inputss[random_number])
                
          #      if torch.rand(1) < 0.6:
               
                    
                   
                inputss[iii] = self.mutation(inputss[iii], gaussian_distribution)
                inputss[iii+n//2] = self.mutation(inputss[iii+n//2], gaussian_distribution)
                inputss[iii+1] = self.mutation(inputss[iii+1], gaussian_distribution)
                inputss[iii+1+n//2] = self.mutation(inputss[iii+1+n//2], gaussian_distribution)
                inputss[iii+2] = self.mutation(inputss[iii+2], gaussian_distribution)
                inputss[iii+2+n//2] = self.mutation(inputss[iii+2+n//2], gaussian_distribution)
                inputss[iii+3] = self.mutation(inputss[iii+3], gaussian_distribution)
                inputss[iii+3+n//2] = self.mutation(inputss[iii+3+n//2], gaussian_distribution)
            
        return inputss
          
       

        
def attention_mechanism(A, B):
    attention_scores = torch.matmul(A, B.T)
    attention_weights = F.softmax(attention_scores, dim=-1)
    k = 40
    random_index = torch.randperm(attention_weights.size(0))[:k]
    selected_rows = attention_weights[random_index]
    max_positions = torch.argmax(selected_rows, dim=1)
    with torch.no_grad():
        for i in range(k):
            fusion_feature = fitness(B, max_positions[i], selected_rows[i])
            A[random_index[i],:] = fusion_feature
    
    return A        
        


           

def fitness(feature, index, weight):
    indices = [
        index,
        index-1 if index % 10 != 0 else None, 
        index+1 if index % 10 != 9 else None, 
        index-10 if index > 9 else None, 
        index+10 if index < 200 else None, 
        index-11 if index > 9 and index % 10 > 0 else None, 
        index-9 if index > 9 and index % 10 < 9 else None, 
        index+9 if index < 200 and index % 10 > 0 else None, 
        index+11 if index < 200 and index % 10 < 9 else None
    ]
    
    indices = [i for i in indices if i is not None]
    feature_values = [feature[i] for i in indices]
    feature_values = torch.stack(feature_values)  #[9,768]
    
    weight_values = [weight[i] for i in indices]
    weight_values = torch.stack(weight_values)  #[9]
    weight_values = F.softmax(weight_values, dim=-1)
    weight_values = weight_values.unsqueeze(1)
    fusion_feature = feature_values * weight_values
    out = torch.sum(fusion_feature, dim=0)
    return out
    
    
    
    
    