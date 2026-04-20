
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), -1.0))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates          = torch.sigmoid(self.gate_scores)   
        pruned_weights = self.weight * gates               
        return F.linear(x, pruned_weights, self.bias)

    @property
    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()



class PrunableNet(nn.Module):


    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_gates(self) -> List[torch.Tensor]:

        return [
            torch.sigmoid(self.fc1.gate_scores),
            torch.sigmoid(self.fc2.gate_scores),
            torch.sigmoid(self.fc3.gate_scores),
        ]

    def sparsity_stats(self, threshold: float = 0.1) -> dict:

        stats = {}
        all_vals = []
        for name, layer in [("fc1", self.fc1), ("fc2", self.fc2), ("fc3", self.fc3)]:
            g = layer.gates.flatten()
            all_vals.append(g)
            pruned = (g < threshold).float().mean().item() * 100
            stats[name] = pruned
        global_gates = torch.cat(all_vals)
        stats["overall"] = (global_gates < threshold).float().mean().item() * 100
        stats["gate_values"] = global_gates.cpu().numpy()
        return stats