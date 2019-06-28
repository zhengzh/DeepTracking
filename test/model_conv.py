
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


class VGGNet(nn.Module):
    
    def __init__(self):

        super(VGGNet, self).__init__()

        self.l1 = nn.Conv2d(4, 64, 3, padding=1)

        self.l2 = nn.Conv2d(64, 128, 3, padding=1)

        self.l3 = nn.Conv2d(128, 256, 3, padding=1)

        self.l4 = nn.Conv2d(256, 512, 3, padding=1)

        self.l5 = nn.Conv2d(512, 1024, 3, padding=1)

        # self.l6 = nn.Conv2d(1024, 2048, 3, padding=1)

        self.pool = nn.MaxPool2d(3, 2, 1)

        # self.out = nn.Linear(4*1024, 2500)
        self.d1 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.d4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.d5 = nn.ConvTranspose2d(64, 6, 3, stride=2, padding=1)



    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        x = self.pool(h1)
        h2 = torch.relu(self.l2(x))
        x = self.pool(h2)
        h3 = torch.relu(self.l3(x))
        x = self.pool(h3)
        h4 = torch.relu(self.l4(x))
        x = self.pool(h4)
        h5 = torch.relu(self.l5(x))
        x = self.pool(h5)
        
        e1 = torch.relu(self.d1(x, output_size=h5.shape))
        e2 = torch.relu(self.d2(e1, output_size=h4.shape))
        e3 = torch.relu(self.d3(e2, output_size=h3.shape))
        e4 = torch.relu(self.d4(e3, output_size=h2.shape))
        e5 = torch.sigmoid(self.d5(e4, output_size=h1.shape))
        
        return e5, [h1, h2, h3, h4]
