import torch.nn as nn
import torch
import torch.nn.functional as F

target = torch.randn(5, requires_grad=True)
input =  torch.randn(5, requires_grad=True)

sig = nn.Sigmoid()
sig_input = sig(input)
print(f'Target:{target}\nInput:{input}\nInput after sigmoid (for BCE):{sig_input}')

# BCE loss
bce_loss = torch.nn.BCELoss()(sig_input, target) # to input edw prepei na einai (0,1)!
print(f'BCE loss:{bce_loss}')

# MSE loss

# an baleis reduction none, anti na gyrnaei to sum h to mean tou (x_n - y_n)^2 tou kathe element, ta gyrnaei ola ws array
mse_no_reduction = F.mse_loss(input, target, reduction="none")
mse_mean = F.mse_loss(input, target, reduction="sum")
mse_sum = F.mse_loss(input, target, reduction="mean")

print(f'MSE for each target-input element pair:{mse_no_reduction}')
print(f'MSE mean:{mse_mean}')
print(f'MSE sum:{mse_sum}')