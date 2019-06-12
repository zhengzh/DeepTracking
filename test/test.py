import torch.nn as nn
import torch
m = nn.Sigmoid()
loss = nn.BCELoss(reduction='sum')
input = torch.randn(3,3,3, requires_grad=True)
target = torch.empty(3,3,3).random_(2)
output = loss(m(input), target)
# output.backward()
output