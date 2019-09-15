
import torch
import torch.nn as nn

class RNNCell(nn.Module):
    
    def __init__(self):

        super(RNNCell, self).__init__()

        k=7
        p=(k-1)//2
        self.embed = nn.Conv2d(1, 16,(k, k), (1,1), (p,p))
        self.hidden = nn.Conv2d(48, 32, (k,k),(1,1),(p,p))
        self.output = nn.Conv2d(32, 1, (k, k), (1,1), (p,p))
    
    def forward(self, x, h):
        x = torch.sigmoid(self.embed(x))
        # x is [batch, channel, width, height]

        x = torch.cat((x, h), 1)
        h = torch.sigmoid(self.hidden(x))
        y = torch.sigmoid(self.output(h))
        
        return h, y

    def forward(self, x, h):
        
        x = torch.relu(self.embed(x))
        # x is [batch, channel, width, height]
        x = torch.cat((x, h), 1)
        h = torch.relu(self.hidden(x))
        y = torch.sigmoid(self.output(h))
        
        return h, y

# hn = 32
hn = 64
# hn = 128

class ConvRNNCell(nn.Module):

    def __init__(self, x, h, dilation):
        super(ConvRNNCell, self).__init__()

        self.l = nn.Conv2d(x+h, h, 3, 1, padding=dilation, dilation=dilation)

    def forward(self, x, h):

        x = torch.cat((x, h), 1)

        return torch.relu(self.l(x))


class ConvRNN(nn.Module):

    def __init__(self):
        super(ConvRNN, self).__init__()

        self.hn = hn = 16
        self.l1 = ConvRNNCell(1, hn, 1)
        self.l2 = ConvRNNCell(hn, hn, 2)
        self.l3 = ConvRNNCell(hn, hn, 4)

        self.out = nn.Conv2d(self.hn, 1, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, X):

        # x is [batch, channel, width, height]
        sh = list(X[0].shape)
        sh[1] = self.hn
        h0 = X[0].new_zeros(sh)
        h3 = h1 = h2 = h0
    
        h_list = []
        y_list = []
        seq_len = X.shape[0]

        for i in range(seq_len):
            
            h0 = self.l1(X[i], h0)
            h1 = self.l2(h0, h1)
            h2 = self.l3(h1, h2)

            out = torch.sigmoid(self.out(h2))

            h_list.append([h0, h1, h2])
            y_list.append(out)

        y = torch.stack(y_list)

        return y, h_list


class ConvRNN2(nn.Module):

    def __init__(self):
        super(ConvRNN2, self).__init__()

        self.hn = hn = 8
        self.l1 = ConvRNNCell(1, hn, 1)
        self.l2 = ConvRNNCell(hn, hn, 2)
        self.l3 = ConvRNNCell(hn, hn, 4)

        self.out = nn.Conv2d(self.hn, 1, (3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, X):

        # x is [batch, channel, width, height]
        sh = list(X[0].shape)
        sh[1] = self.hn
        h0 = X[0].new_zeros(sh)
        h3 = h1 = h2 = h0
    
        h_list = []
        y_list = []
        seq_len = X.shape[0]

        for i in range(seq_len):
            
            h0 = self.l1(X[i], h0)
            h1 = self.l2(h0, h1)
            h2 = self.l3(h1, h2)

            out = torch.sigmoid(self.out(h2))

            h_list.append([h0, h1, h2])
            y_list.append(out)

        y = torch.stack(y_list)

        return y, h_list

class RNNCell2(nn.Module):
    
    def __init__(self):

        super(RNNCell2, self).__init__()

        k=7
        p=(k-1)//2
        self.l1 = nn.Conv2d(hn+1, hn,(3, 3), (1,1), (1,1), (1, 1))
        self.l2 = nn.Conv2d(2*hn, hn, (3,3),(1,1),(2, 2), dilation=(2, 2))
        self.l3 = nn.Conv2d(2*hn, hn, (3,3),(1,1),(4,4), dilation=(4, 4))
        self.output = nn.Conv2d(hn, 1, (3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, x, h0, h1, h2):
        
        # x is [batch, channel, width, height]
        x = torch.cat((x, h0), 1)
        h0 = torch.relu(self.l1(x))
        x = torch.cat((h0, h1), 1)
        h1 = torch.relu(self.l2(x))
        x = torch.cat((h1, h2), 1)
        h2 = torch.relu(self.l3(x))

        y = torch.sigmoid(self.output(h2))
        
        return h0, h1, h2, y
        

class RNNCell3(nn.Module):
    
    def __init__(self):

        super(RNNCell3, self).__init__()

        k=7
        p=(k-1)//2
        self.l1 = nn.Conv2d(hn+1, hn,(3, 3), (1,1), (1,1), (1, 1))
        self.l2 = nn.Conv2d(2*hn, hn, (3,3),(1,1),(2, 2), dilation=(2, 2))
        self.l3 = nn.Conv2d(2*hn, hn, (3,3),(1,1),(4,4), dilation=(4, 4))
        self.output = nn.Conv2d(3*hn, 1, (3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, x, h0, h1, h2):
        
        # x is [batch, channel, width, height]
        # (h0',x)->h0
        # (h1',h0)->h1
        # (h2',h1)->h2
        # (h0,h1,h2)->y
        x = torch.cat((x, h0), 1)
        h0 = torch.relu(self.l1(x))
        x = torch.cat((h0, h1), 1)
        h1 = torch.relu(self.l2(x))
        x = torch.cat((h1, h2), 1)
        h2 = torch.relu(self.l3(x))

        h_cat = torch.cat((h0, h1, h2), 1)
        y = torch.sigmoid(self.output(h_cat))
        
        return h0, h1, h2, y

class GRUCell(nn.Module):
    
    def __init__(self, dil, xn):

        super(GRUCell, self).__init__()

        hn = 8
        self.z = nn.Conv2d(hn+xn, hn,(3, 3), (1,1), (dil,dil), dilation=(dil, dil), bias=True)
        self.r = nn.Conv2d(hn+xn, hn,(3, 3), (1,1), (dil,dil), dilation=(dil, dil), bias=True)
        self.h = nn.Conv2d(hn+xn, hn,(3, 3), (1,1), (dil,dil), dilation=(dil, dil), bias=True)
    
    
    def forward(self, x, h):
        
        # x is [batch, channel, width, height]

        hx = torch.cat((x, h), 1)
        z = torch.sigmoid(self.z(hx))
        r = torch.sigmoid(self.r(hx))
        h1 = torch.cat((r*h, x), 1)
        h1 = torch.tanh(self.h(h1))
        
        o = (1-z)*h + z * h1
        
        return o


class GRURNN(nn.Module):
    
    def __init__(self):

        super(GRURNN, self).__init__()

        self.hn = hn = 8

        self.l1 = GRUCell(1, 1)
        self.l2 = GRUCell(2, self.hn)
        self.l3 = GRUCell(4, self.hn)
        # self.l4 = GRUCell(, self.hn, 27)

        self.out1 = nn.Conv2d(self.hn, hn, 3, padding=2, dilation=2)
        self.out2 = nn.Conv2d(self.hn, hn, 3, padding=1, dilation=1)

        self.out = nn.Conv2d(self.hn, 1, (3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, X):
        
        # x is [batch, channel, width, height]
        sh = list(X[0].shape)
        sh[1] = self.hn
        h0 = X[0].new_zeros(sh)
        h3 = h1 = h2 = h0
    
        h_list = []
        y_list = []
        seq_len = X.shape[0]

        for i in range(seq_len):

            h0 = self.l1(X[i], h0)
            h1 = self.l2(h0, h1)
            h2 = self.l3(h1, h2)
            # h3 = self.l4(h2, h3)
            
            y = torch.relu(self.out1(h2))
            y = torch.relu(self.out2(y))
            # y = torch.relu(self.out3(y))
            y = torch.sigmoid(self.out(y))
            h_list.append([h0, h1, h2, h3])
            y_list.append(y)

        y = torch.stack(y_list)
        return y, h_list


class RNN2(nn.Module):
    
    def __init__(self, width, height):
        super(RNN2, self).__init__()

        self.width = width
        self.height = height
        self.rnn = RNNCell2()
        
        # self.register_buffer('h', torch.zeros(10, hn, self.width, self.height))
    
    def forward(self, X):
    
        # X[seq_len, batch, channel, width, height]
        # b = X.shape[1]
        # h0 = self.h[:b]

        sh = list(X[0].shape)
        sh[1] = hn
        h0 = X[0].new_zeros(sh)

        h1 = h0
        h2 = h0
        h_list = []
        y_list = []
        seq_len = X.shape[0]
        for i in range(seq_len):
            h0, h1, h2, y = self.rnn(X[i], h0, h1, h2)
            y_list.append(y)
            h_list.append((h0, h1, h2))

        return torch.stack(y_list), h_list

class RNN(nn.Module):
    
    def __init__(self, width, height):
        super(RNN, self).__init__()

        self.width = width
        self.height = height
        self.rnn = RNNCell()
        
        # self.register_buffer('h', torch.zeros(10, 32, self.width, self.height))
    
    def forward(self, X):
    
        # X[seq_len, batch, channel, width, height]
        # b = X.shape[1]
        # h = torch.zeros(1, 32, self.width, self.height)
        # h = self.h[:b]
        
        h = X[0].clone()
        h.zero_()

        h_list = []
        y_list = []
        seq_len = X.shape[0]
        for i in range(seq_len):
            h, y = self.rnn(X[i], h)
            y_list.append(y)
            h_list.append(h)

        return torch.stack(y_list), torch.stack(h_list)
