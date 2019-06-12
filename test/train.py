
import torch.optim as optim
import torch
import torch.nn as nn

from model import RNN
import numpy as np



epochs = 1000

criterion = nn.BCELoss(reduction='sum')


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data = np.load('./save/data2.npy')

n, l, w, h = data.shape
model = RNN(w, h)
model.to(device)

data = torch.from_numpy(data).float()


model_optim = optim.Adam(model.parameters(), lr=0.001)

index = np.arange(n)


def getSequence(i):
    input = data[i].unsqueeze(1).unsqueeze(1).to(device)
    return input

def dropotuInput(target):
    input = target.clone()
    for i, image in enumerate(input):
        if (i-1) % 10 >= 5:
            image.zero_()

    return input


def evaluate(weights):
    input = getSequence(1)

    model(input)
    pass


def train():
    target = getSequence(np.random.randint(n))
    input = dropotuInput(target)
    
    output, hidden = model(input)

    loss = criterion(output, target)

    model_optim.zero_grad()
    loss.backward()
    model_optim.step() 
    
    return loss.item()

total_cost = 0

import os

os.makedirs('./save/weights', exist_ok=True)
os.makedirs('.save/video', exist_ok=True)

for k in range(1, epochs+1):

    cost = train()

    total_cost += cost

    if k % 100 == 0:
        print('Iteration % d, cost: %f' % (k, total_cost/1000))
        total_cost = 0
        
        torch.save(model.state_dict(),  './save/weights/%d.dat' % (k))
