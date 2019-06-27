
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
        self.output = nn.Conv2d(3*h, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output1 = nn.Conv2d(64, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))



    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))

        cat = torch.cat((h1, h2, h3), 1)

        y = torch.relu(self.output(cat))
        y = torch.sigmoid(self.output1(y))

        return y, [h1, h2, h3]
