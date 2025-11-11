import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpUtilityNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=0, hidden_units=8):
        super(MlpUtilityNetwork, self).__init__()
        self.input_norm = nn.LayerNorm(input_size, elementwise_affine=True)
        layers = []
        if hidden_layers > 0:
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(nn.ReLU())
            for _ in range(hidden_layers - 1):
                layers.append(nn.Linear(hidden_units, hidden_units))
                layers.append(nn.BatchNorm1d(hidden_units))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_units, output_size))
            self.network = nn.Sequential(*layers)
        else:
            self.network = nn.Linear(input_size, output_size)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        #x = (x - x.mean(dim=0)) / (x.std(dim=0)+1e-6)
        x = self.input_norm(x)
        x=self.network(x)
        return x