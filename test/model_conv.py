
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
        self.output1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output2 = nn.Conv2d(64, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))



    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))
        h4 = torch.relu(self.l4(h3))

        cat = torch.cat((h1, h2, h3, h4), 1)
        # cat = h1 + h3

        y = torch.relu(self.output(cat))
        y = torch.relu(self.output1(y))
        y = torch.sigmoid(self.output2(y))

        return y, [h1, h2, h3, h4]


class Net4(nn.Module):
    
    def __init__(self):

        super(Net4, self).__init__()

        h = 16
        dil = [1, 3, 9, 27, 9, 3]
        # cha = [16, 64, 64, 64, 64, 64]
        cha = [16, 16, 16, 16, 16, 16]
        # self.l1 = nn.Conv2d(10, cha[0], (3, 3), (1, 1), (1, 1), dilation=(1,1))
        # self.l2 = nn.Conv2d(cha[0], cha[1], (3, 3), (1, 1), (d2, d2), dilation=(d2,d2))
        # self.l3 = nn.Conv2d(cha[1], cha[2], (3, 3), (1, 1), (d3, d3), dilation=(d3,d3))
        # self.l4 = nn.Conv2d(cha[2], cha[3], (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))
        # self.l4 = nn.Conv2d(cha[2], cha[3], (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))

        self.l1 = self.conv3(10, cha[0], dil[0])
        self.l2 = self.conv3(cha[0], cha[1], dil[1])
        self.l3 = self.conv3(cha[1], cha[2], dil[2])
        self.l4 = self.conv3(cha[2], cha[3], dil[3])
        self.l5 = self.conv3(cha[3], cha[4], dil[4])
        self.l6 = self.conv3(cha[4], cha[5], dil[5])
        
        # input_channel, output_channel, kernel, stride, padding, dialation
        self.output = nn.Conv2d(96, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output2 = nn.Conv2d(64, 10, (3, 3), (1, 1), (1, 1), dilation=(1,1))


    def conv3(self, inplanes, planes, dilation):
        return nn.Conv2d(inplanes, planes, 3, 1, padding=dilation, dilation=dilation)

    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))
        h4 = torch.relu(self.l4(h3))
        h5 = torch.relu(self.l5(h4))
        h6 = torch.relu(self.l6(h5))

        cat = torch.cat((h1, h2, h3, h4, h5, h6), 1)
        # cat = h1 + h3

        # cat = h4
        y = torch.relu(self.output(cat))
        y = torch.relu(self.output1(y))
        y = torch.sigmoid(self.output2(y))

        return y, [h1, h2, h3, h4]


class Net6(nn.Module):
    
    def __init__(self):

        super(Net6, self).__init__()

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
        self.output1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output2 = nn.Conv2d(64, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))



    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))
        h4 = torch.relu(self.l4(h3))

        cat = torch.cat((h1, h2, h3, h4), 1)
        # cat = h1 + h3

        y = torch.relu(self.output(cat))
        y = torch.relu(self.output1(y))
        y = torch.sigmoid(self.output2(y))

        return y, [h1, h2, h3, h4]

BatchNorm = nn.BatchNorm2d

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, bias=False),
                BatchNorm(planes),
            )

        self.downsample = downsample
        
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class NetDRN(nn.Module):

    def __init__(self):

        super(NetDRN, self).__init__()

        h = 64

        self.conv1 = nn.Conv2d(10, h, 3, stride=1, padding=1)
        self.l1 = BasicBlock(h, h, dilation=(1, 1))
        self.l2 = BasicBlock(h, h, dilation=(2, 2))
        self.l3 = BasicBlock(h, h, dilation=(4, 4))
        self.l4 = BasicBlock(h, h, dilation=(8, 8))
        self.l5 = BasicBlock(h, h, dilation=(16, 16))
        # self.l6 = BasicBlock(h, h, dilation=(32, 32))
        # self.l7 = BasicBlock(h, h, dilation=(16, 16))
        self.l8 = BasicBlock(h, h, dilation=(8, 8))
        self.l9 = BasicBlock(h, h, dilation=(4, 4))
        self.l10 = BasicBlock(h, h, dilation=(2, 2))
        self.o1 = nn.Conv2d(h, h, 3, stride=1, padding=1)
        self.o2 = nn.Conv2d(h, 10, 3, stride=1, padding=1)

    def forward(self, x):

        x = torch.relu(self.conv1(x))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        x = torch.relu(self.o1(x))
        x = torch.sigmoid(self.o2(x))

        return x, None

class NetDRN2(nn.Module):

    def __init__(self):

        super(NetDRN2, self).__init__()

        h = 64

        self.conv1 = nn.Conv2d(10, 16, 3, stride=1, padding=1)
        self.l1 = BasicBlock(16, 32, dilation=(1, 1))
        self.l2 = BasicBlock(32, 48, dilation=(3, 3))
        self.l3 = BasicBlock(48, 64, dilation=(9, 9))
        self.l4 = BasicBlock(64, 96, dilation=(27, 27))
        self.l5 = BasicBlock(96, 128, dilation=(9, 9))
        self.l6 = BasicBlock(128, h, dilation=(3, 3))
        # self.l6 = BasicBlock(h, h, dialation=(16, 32))
        self.o1 = nn.Conv2d(h, h, 3, stride=1, padding=1)
        self.o2 = nn.Conv2d(h, 10, 3, stride=1, padding=1)

    def forward(self, x):

        x = torch.relu(self.conv1(x))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        x = torch.relu(self.o1(x))
        x = torch.sigmoid(self.o2(x))

        return x, None


class NetDRN5(nn.Module):

    def __init__(self):

        super(NetDRN2, self).__init__()

        h = 64

        self.conv1 = nn.Conv2d(10, 16, 3, stride=1, padding=1)
        self.l1 = BasicBlock(16, 32, dilation=(1, 1))
        self.l2 = BasicBlock(32, 48, dilation=(1, 1))
        self.l3 = BasicBlock(48, 64, dilation=(9, 9))
        self.l4 = BasicBlock(64, 96, dilation=(27, 27))
        self.l5 = BasicBlock(96, 128, dilation=(9, 9))
        self.l6 = BasicBlock(128, h, dilation=(3, 3))
        # self.l6 = BasicBlock(h, h, dialation=(16, 32))
        self.o1 = nn.Conv2d(h, h, 3, stride=1, padding=1)
        self.o2 = nn.Conv2d(h, 10, 3, stride=1, padding=1)

    def forward(self, x):

        x = torch.relu(self.conv1(x))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        x = torch.relu(self.o1(x))
        x = torch.sigmoid(self.o2(x))

        return x, None


class NetDRN3(nn.Module):

    def __init__(self):

        super(NetDRN3, self).__init__()

        h = 64

        self.conv1 = nn.Conv2d(10, h, 3, stride=1, padding=1)
        self.l1 = BasicBlock(h, h, dilation=(1, 1))
        self.l2 = BasicBlock(h, h, dilation=(3, 3))
        self.l3 = BasicBlock(h, h, dilation=(9, 9))
        self.l4 = BasicBlock(h, h, dilation=(27, 27))
        self.l5 = BasicBlock(h, h, dilation=(27, 27))
        self.l6 = BasicBlock(h, h, dilation=(27, 27))

        self.o1 = nn.Conv2d(h, h, 3, stride=1, padding=1)
        self.o2 = nn.Conv2d(h, 10, 3, stride=1, padding=1)

    def forward(self, x):

        x = torch.relu(self.conv1(x))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        x = torch.relu(self.o1(x))
        x = torch.sigmoid(self.o2(x))

        return x, None

class NetDRN4(nn.Module):

    def __init__(self):

        super(NetDRN4, self).__init__()

        self.conv1 = nn.Conv2d(10, 16, 3, stride=1, padding=1)
        self.l1 = BasicBlock(16, 16, dilation=(1, 1))
        self.l2 = BasicBlock(16, 16, dilation=(3, 3))
        self.l3 = BasicBlock(16, 64, dilation=(9, 9))
        self.l4 = BasicBlock(64, 128, dilation=(27, 27))

        self.o1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.o2 = nn.Conv2d(128, 10, 3, stride=1, padding=1)

    def forward(self, x):

        x = torch.relu(self.conv1(x))

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)

        x = torch.relu(self.o1(x))
        x = torch.sigmoid(self.o2(x))

        return x, None


class NetTestDilation(nn.Module):
    
    # same result as small dilation
    def __init__(self):

        super(NetTestDilation, self).__init__()

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
        self.l2 = nn.Conv2d(h, h, (3, 3), (1, 1), (d2, d2), dilation=(d2,d2))
        self.l3 = nn.Conv2d(h, h, (3, 3), (1, 1), (d3, d3), dilation=(d3,d3))
        self.l4 = nn.Conv2d(h, h, (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))
        
        # input_channel, output_channel, kernel, stride, padding, dialation
        self.output = nn.Conv2d(4*h, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output2 = nn.Conv2d(64, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))



    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))
        h4 = torch.relu(self.l4(h3))

        cat = torch.cat((h1, h2, h3, h4), 1)
        # cat = h1 + h3

        y = torch.relu(self.output(cat))
        y = torch.relu(self.output1(y))
        y = torch.sigmoid(self.output2(y))

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
        self.l2 = nn.Conv2d(h, h, (3, 3), (1, 1), (d2, d2), dilation=(d2,d2))
        self.l3 = nn.Conv2d(h, h, (3, 3), (1, 1), (d3, d3), dilation=(d3,d3))
        self.l4 = nn.Conv2d(h, h, (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))
        
        # input_channel, output_channel, kernel, stride, padding, dialation
        self.output = nn.Conv2d(4*h, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output2 = nn.Conv2d(64, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))



    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))
        h4 = torch.relu(self.l4(h3))

        cat = torch.cat((h1, h2, h3, h4), 1)
        # cat = h1 + h3

        y = torch.relu(self.output(cat))
        y = torch.relu(self.output1(y))
        y = torch.sigmoid(self.output2(y))

        return y, [h1, h2, h3, h4]

class NetMulHead(nn.Module):
    
    # same result as small dilation
    def __init__(self):

        super(NetMulHead, self).__init__()

        h = 32
        d1 = 1
        d2 = 3
        d3 = 9
        d4 = 27
        self.l1 = nn.Conv2d(4, h, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.l2 = nn.Conv2d(h, h, (3, 3), (1, 1), (d2, d2), dilation=(d2,d2))
        self.l3 = nn.Conv2d(h, h, (3, 3), (1, 1), (d3, d3), dilation=(d3,d3))
        self.l4 = nn.Conv2d(h, h, (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))
        
        # input_channel, output_channel, kernel, stride, padding, dialation

        outputs = []
        outputs1 = []
        for i in range(6):
            outputs.append(nn.Conv2d(h, h, (3, 3), (1, 1), (1, 1), dilation=(1,1)))
            outputs1.append(nn.Conv2d(h, 1, (3, 3), (1, 1), (1, 1), dilation=(1,1)))

        self.outputs = nn.ModuleList(outputs)
        self.outputs1 = nn.ModuleList(outputs1)


    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))
        h4 = torch.relu(self.l4(h3))

        # cat = torch.cat((h1, h2, h3, h4), 1)
        # cat = h1 + h3

        cat = h4

        ys = []
        for i in range(6):
            y = torch.relu(self.outputs[i](cat))
            y = torch.sigmoid(self.outputs1[i](y))
            ys.append(y)

        y = torch.cat(ys, 1)

        return y, [h1, h2, h3, h4]

class NetGridArtifact(nn.Module):
    
    # same result as small dilation
    def __init__(self):

        super(NetGridArtifact, self).__init__()

        h = 64
        d1 = 1
        d2 = 3
        d3 = 9
        d4 = 27
        self.l1 = nn.Conv2d(4, h, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.l2 = nn.Conv2d(h, h, (3, 3), (1, 1), (d2, d2), dilation=(d2,d2))
        self.l3 = nn.Conv2d(h, h, (3, 3), (1, 1), (d3, d3), dilation=(d3,d3))
        self.l4 = nn.Conv2d(h, h, (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))
        self.l5 = nn.Conv2d(h, h, (3, 3), (1, 1), d3, dilation=(d3,d3))
        self.l6 = nn.Conv2d(h, h, (3, 3), (1, 1), d2, dilation=(d2,d2))
        self.l7 = nn.Conv2d(h, h, (3, 3), (1, 1), d1, dilation=(d1,d1))
        
        # input_channel, output_channel, kernel, stride, padding, dialation
        self.output = nn.Conv2d(h, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output1 = nn.Conv2d(64, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))



    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        h3 = torch.relu(self.l3(h2))
        h4 = torch.relu(self.l4(h3))
        h = torch.relu(self.l5(h4))
        h = torch.relu(self.l6(h))
        h = torch.relu(self.l7(h))

        # cat = torch.cat((h1, h2, h3, h4), 1)
        # cat = h1 + h3

        cat = h
        y = torch.relu(self.output(cat))
        y = torch.sigmoid(self.output1(y))

        return y, [h1, h2, h3, h4]


class Net2(nn.Module):
    
    def __init__(self):

        super(Net2, self).__init__()

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


class Net3(nn.Module):
    
    def __init__(self):

        super(Net3, self).__init__()

        h = 64
        d1 = 1
        d2 = 3
        d3 = 9
        d4 = 27
        self.l1 = nn.Conv2d(4, h, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.l11 = nn.Conv2d(h, h, 1)
        self.l12 = nn.Conv2d(h, h, 1)
        self.l2 = nn.Conv2d(h, h, (3, 3), (1, 1), (d2, d2), dilation=(d2,d2))
        self.l21 = nn.Conv2d(h, h, 1)
        self.l22 = nn.Conv2d(h, h, 1)

        self.l3 = nn.Conv2d(h, h, (3, 3), (1, 1), (d3, d3), dilation=(d3,d3))
        self.l31 = nn.Conv2d(h, h, 1)
        self.l32 = nn.Conv2d(h, h, 1)

        self.l4 = nn.Conv2d(h, h, (3, 3), (1, 1), (d4, d4), dilation=(d4,d4))
        self.l41 = nn.Conv2d(h, h, 1)
        self.l42 = nn.Conv2d(h, h, 1)
        
        # input_channel, output_channel, kernel, stride, padding, dialation
        self.output = nn.Conv2d(4*h, 64, (3, 3), (1, 1), (1, 1), dilation=(1,1))
        self.output1 = nn.Conv2d(64, 6, (3, 3), (1, 1), (1, 1), dilation=(1,1))



    def forward(self, x):

        h1 = torch.relu(self.l1(x))
        h1 = torch.relu(self.l11(h1))
        h1 = torch.relu(self.l12(h1))
        h2 = torch.relu(self.l2(h1))
        h2 = torch.relu(self.l21(h2))
        h2 = torch.relu(self.l22(h2))
        h3 = torch.relu(self.l3(h2))
        h3 = torch.relu(self.l31(h3))
        h3 = torch.relu(self.l32(h3))
        h4 = torch.relu(self.l4(h3))
        h4 = torch.relu(self.l41(h4))
        h4 = torch.relu(self.l42(h4))

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
