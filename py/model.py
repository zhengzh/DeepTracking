import torch
import torch.nn as nn



class RNNCell(nn.Module):
    
    def __init__(self):

        super(RNNCell, self).__init__()

        self.embed = nn.Conv2d(2, 16,(7, 7), (1,1), (3,3))
        self.hidden = nn.Conv2d(48, 32, (7,7),(1,1),(3,3))
        self.output = nn.Conv2d(32, 1, (7, 7), (1,1), (3,3))
    
    def forward(self, x, h):
        x = self.embed(x)
        #[batch, channel, width, height]
        x = torch.cat(x, h, 1)
        h = self.hidden(x)
        y = self.output(h)
        return h, y
        

class RNN(nn.Module):
    
    def __init__(self, width, height):

        super(RNN, self).__init__()

        self.rnn = RNNCell()
    
    def forward(self, X):
        
        h = torch.zeros(32, width, height)
        h_list = []
        y_list = []
        seq_len = X.shape[0]
        for i in range(seq_len):
            h, y = self.rnn(X[i], h)
            y_list.append(y)
            h_list.append(h)
        
            return torch.stack(y_list), torch.stack(h_list)

criterion = nn.BCELoss()

#%% 
rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
#%% [markdown]
# hello