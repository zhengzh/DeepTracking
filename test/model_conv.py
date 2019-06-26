
import torch
import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self):

        super(Net, self).__init__()

        h = 64
        self.l1 = nn.Conv2d(4, h, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.l2 = nn.Conv2d(h, h, (3, 3), (1, 1), (3, 3), dilation=(3,3))
        self.l3 = nn.Conv2d(h, h, (3, 3), (1, 1), (9, 9), dilation=(9,9))
        
        # input_channel, output_channel, kernel, stride, padding, dialation
        self.output = nn.Conv2d(h, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))


    def forward(self, x):

        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.sigmoid(self.output(x))

        return x
