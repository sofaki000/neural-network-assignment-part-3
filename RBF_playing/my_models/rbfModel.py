import torch.nn as nn
import torch

def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

class MyRbfModel(nn.Module):
    def __init__(self, output_features, input_features):
        super(MyRbfModel, self).__init__()
        self.output_features=output_features
        self.input_features =input_features
        self.centers = nn.Parameter(torch.Tensor(output_features, input_features))
        self.basis_function = gaussian
    def forward(self, input):
        size = (input.size(0), self.output_features, self.input_features)
