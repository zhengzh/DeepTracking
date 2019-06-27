import torch
import torch.nn as nn

class GRUCell(nn.Module):
    
    def __init__(self, dil, xn, pad):

        super(GRUCell, self).__init__()

        hn = 32
        self.z = nn.Conv2d(hn+xn, hn,(3, 3), (1,1), (pad,pad), dilation=(dil, dil))
        self.r = nn.Conv2d(hn+xn, hn,(3, 3), (1,1), (pad,pad), dilation=(dil, dil))
        self.h = nn.Conv2d(hn+xn, hn,(3, 3), (1,1), (pad,pad), dilation=(dil, dil))
    
    
    def forward(self, x, h):
        
        # x is [batch, channel, width, height]

        hx = torch.cat((x, h), 1)
        z = torch.sigmoid(self.z(hx))
        r = torch.sigmoid(self.r(hx))
        h1 = torch.cat((r*h, x), 1)
        h1 = torch.tanh(self.h(h1))
        
        o = (1-z)*h + z * h1
        
        return o