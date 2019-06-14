
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

class RNNCell2(nn.Module):
    
    def __init__(self):

        super(RNNCell2, self).__init__()

        k=7
        p=(k-1)//2
        self.l1 = nn.Conv2d(17, 16,(3, 3), (1,1), (1,1), (1, 1))
        self.l2 = nn.Conv2d(32, 16, (3,3),(1,1),(2, 2), dilation=(2, 2))
        self.l3 = nn.Conv2d(32, 16, (3,3),(1,1),(4,4), dilation=(4, 4))
        self.output = nn.Conv2d(16, 1, (3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, x, h0, h1, h2):
        
        x = torch.cat((x, h0), 1)
    
        h0 = torch.sigmoid(self.l1(x))
        # x is [batch, channel, width, height]

        x = torch.cat((h0, h1), 1)
        h1 = torch.sigmoid(self.l2(x))
        x = torch.cat((h1, h2), 1)
        h2 = torch.sigmoid(self.l3(x))

        y = torch.sigmoid(self.output(h2))
        
        return h0, h1, h2, y



class RNN2(nn.Module):
    
    def __init__(self, width, height):
        super(RNN2, self).__init__()

        self.width = width
        self.height = height
        self.rnn = RNNCell2()
        
        self.register_buffer('h', torch.zeros(10, 16, self.width, self.height))
    
    def forward(self, X):
    
        # X[seq_len, batch, channel, width, height]
        b = X.shape[1]
        h0 = self.h[:b]
        h1 = h0
        h2 = h0
        h_list = []
        y_list = []
        seq_len = X.shape[0]
        for i in range(seq_len):
            h0, h1, h2, y = self.rnn(X[i], h0, h1, h2)
            y_list.append(y)
            # h_list.append((h0, h1, h2))

        return torch.stack(y_list), []

class RNN(nn.Module):
    
    def __init__(self, width, height):
        super(RNN, self).__init__()

        self.width = width
        self.height = height
        self.rnn = RNNCell()
        
        self.register_buffer('h', torch.zeros(10, 32, self.width, self.height))
    
    def forward(self, X):
    
        # X[seq_len, batch, channel, width, height]
        b = X.shape[1]
        # h = torch.zeros(1, 32, self.width, self.height)
        h = self.h[:b]

        h_list = []
        y_list = []
        seq_len = X.shape[0]
        for i in range(seq_len):
            h, y = self.rnn(X[i], h)
            y_list.append(y)
            h_list.append(h)

        return torch.stack(y_list), torch.stack(h_list)
