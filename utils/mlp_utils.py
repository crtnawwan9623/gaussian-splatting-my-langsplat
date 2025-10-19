import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpUtilityNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=3, hidden_units=128):
        super(MlpUtilityNetwork, self).__init__()
        layers = []
        if hidden_layers > 0:
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_units, output_size))
            self.network = nn.Sequential(*layers)
        else:
            self.network = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x=self.network(x)
        return F.softmax(x, dim=-1)