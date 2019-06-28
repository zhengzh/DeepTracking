
import torch
import torch.nn as nn

class Net(nn.Module):
    
    def __init__(self):

        super(Net, self).__init__()

        h = 64
        d1 = 1
        d2 = 3
        d3 = 9
        d4 = 27
        self.l1 = nn.Conv2d(4, h, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.l2 = nn.Conv2d(h, h, (3, 3), (1, 1), (d2, d2), dilation=(d2,d2))
        self.l3 = nn.Conv2d(h, h, (3, 3), (1, 1), (d3, d3), dilation=(d3,d3))
        self.l4 = nn.Conv2d(h, h, (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))
        
        # input_channel, output_channel, kernel, stride, padding, dialation
        self.output = nn.Conv2d(4*h, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output1 = nn.Conv2d(64, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))



    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))
        h4 = torch.relu(self.l4(h3))

        cat = torch.cat((h1, h2, h3, h4), 1)
        # cat = h1 + h3

        y = torch.relu(self.output(cat))
        y = torch.sigmoid(self.output1(y))

        return y, [h1, h2, h3, h4]


class Net(nn.Module):
    
    def __init__(self):

        super(Net, self).__init__()

        h = 64
        d1 = 1
        d2 = 3
        d3 = 9
        d4 = 27
        self.l1 = nn.Conv2d(4, h, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.l11 = nn.Conv2d(h, h, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.l2 = nn.Conv2d(h, h, (3, 3), (1, 1), (d2, d2), dilation=(d2,d2))
        self.l21 = nn.Conv2d(h, h, (3, 3), (1, 1), (d2, d2), dilation=(d2,d2))
        self.l3 = nn.Conv2d(h, h, (3, 3), (1, 1), (d3, d3), dilation=(d3,d3))
        self.l31 = nn.Conv2d(h, h, (3, 3), (1, 1), (d3, d3), dilation=(d3,d3))
        self.l4 = nn.Conv2d(h, h, (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))
        self.l41 = nn.Conv2d(h, h, (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))
        
        # input_channel, output_channel, kernel, stride, padding, dialation
        self.output = nn.Conv2d(4*h, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output1 = nn.Conv2d(64, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))



    def forward(self, x):
        m = torch.relu
        h1 = m(self.l1(x))
        h1 = m(self.l11(h1))
        h2 = m(self.l2(h1))
        h2 = m(self.l21(h2))
        h3 = m(self.l3(h2))
        h3 = m(self.l31(h3))
        h4 = m(self.l4(h3))
        h4 = m(self.l41(h4))

        cat = torch.cat((h1, h2, h3, h4), 1)
        # cat = h1 + h3

        y = torch.relu(self.output(cat))
        y = torch.sigmoid(self.output1(y))

        return y, [h1, h2, h3, h4]