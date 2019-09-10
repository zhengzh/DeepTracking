import torch.optim as optim
import torch
import torch.nn as nn

# from model_conv import NetTestDilation as Net
from model_conv import NetMulHead as Net
# from model_conv import NetGridArtifact as Net
# from model_conv import Net2 as Net
# from model_conv import NetDRN as Net
import numpy as np


epochs = 1000

criterion = nn.BCELoss(reduction='sum')


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

orig_data = np.load('./save/data2.npy')

def process_data(data):
    n, l, w, h = data.shape
    res = []
    for i in range(0,l,10):
        res.append(data[:,i:i+10])
    
    res = np.concatenate(res, axis=0)
    
    return res

data = process_data(orig_data)

n, l, w, h = data.shape
model = Net()

# if torch.cuda.device_count() > 0:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model)

# weight = torch.load('./save/weights/100000.dat')
# model.load_state_dict(weight)

model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('total parameters: %d' % count_parameters(model))


data = torch.from_numpy(data).float()


model_optim = optim.Adam(model.parameters(), lr=0.00001)
# model_optim = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

index = np.arange(n)

data = data.to(device)

def getSequence(i):
    
    x = data[i, :4]
    y = data[i, 4:]

    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    return x, y


from PIL import Image

import matplotlib.pyplot as plt

def img_grey(data):
    data = data.astype(np.uint8)
    return Image.fromarray(data * 255, mode='L').convert('1')

def evaluate(weights,idx=0):
    input, target = getSequence(idx)

    model.load_state_dict(weights)

    with torch.no_grad():
        output, hidden = model(input)
    
    target = target.squeeze()
    input = input.squeeze()
    output = output.squeeze()

    target = target.cpu().numpy()
    input = input.cpu().numpy()
    output = output.cpu().numpy()

    n, h, w = target.shape

    for i in range(n):
        x = img_grey(target[i]>0.5)
        y = img_grey(output[i]>0.5)

        x.save('./save/video/input_%i.bmp' % (i))
        y.save('./save/video/output_%i.bmp' % (i))


def train():
    x, y = getSequence(np.random.randint(n))
  
    output, hidden = model(x)

    loss = criterion(output, y)

    model_optim.zero_grad()
    loss.backward()
    model_optim.step() 
    
    return loss.item()


import os

os.makedirs('./save/weights', exist_ok=True)
os.makedirs('./save/video', exist_ok=True)

from tqdm import tqdm

def main(args):
    total_cost = 0

    log = 1000
    epochs = args.epochs

    avg_cost = []

    for k in tqdm(range(1, epochs+1)):

        cost = train()

        total_cost += cost

        if k % log == 0:
            avg_cost.append(total_cost / log)
            print('Iteration % d, cost: %f' % (k, total_cost/log))
            total_cost = 0
            plt.plot(avg_cost)
            plt.savefig('loss.png')
            
            evaluate(model.state_dict())
            torch.save(model.state_dict(),  './save/weights/%d.dat' % (k))


def test():
    weight = torch.load('./save/weights/100000.dat')
    evaluate(weight,idx=1)
    pass

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating moving ball')
    parser.add_argument('--epochs', type=int, default=100000)
    args = parser.parse_args()
    main(args)
    #test()