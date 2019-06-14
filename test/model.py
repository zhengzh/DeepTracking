
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
        

class RNN(nn.Module):
    
    def __init__(self, width, height):
        super(RNN, self).__init__()

        self.width = width
        self.height = height
        self.rnn = RNNCell()
        
        self.register_buffer('h', torch.zeros(1, 32, self.width, self.height))
    
    def forward(self, X):
    
        # X[seq_len, batch, channel, width, height]
        # b = X[1]
        # h = torch.zeros(1, 32, self.width, self.height)
        h = self.h

        h_list = []
        y_list = []
        seq_len = X.shape[0]
        for i in range(seq_len):
            h, y = self.rnn(X[i], h)
            y_list.append(y)
            h_list.append(h)

        return torch.stack(y_list), torch.stack(h_list)
