
import torch.optim as optim
import torch
import torch.nn as nn

from model import ConvRNN as RNN
# from model import GRURNN as RNN
import numpy as np


epochs = 1000

criterion = nn.BCELoss(reduction='sum')


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data = np.load('./save/data2.npy')

n, l, w, h = data.shape
model = RNN()

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


model_optim = optim.Adam(model.parameters(), lr=0.001)

index = np.arange(n)

# data = data.to(device).unsqueeze(2).unsqueeze(2)

def getSequence(i):
    
    input = data[i]
    input = input.to(device).unsqueeze(1).unsqueeze(1)
    return input


def dropoutInput(target, interval=20):
    input = target.clone()
    for i, image in enumerate(input):
        # if (i) % interval >= interval/2.0:
        if i >= interval/2.0:
            image.zero_()

    return input

from PIL import Image

import matplotlib.pyplot as plt

def img_grey(data):
    data = data.astype(np.uint8)
    return Image.fromarray(data * 255, mode='L').convert('1')

def evaluate(weights,idx=0):
    target = getSequence(idx)
    input = dropoutInput(target)


    model.load_state_dict(weights)

    with torch.no_grad():
        output, hidden = model(input)
    
    target = target.squeeze()
    input = input.squeeze()
    output = output.squeeze()

    target = target.cpu().numpy()
    input = input.cpu().numpy()
    output = output.cpu().numpy()

    n, h, w = input.shape

    for i in range(n):
        x = img_grey(target[i]>0.5)
        y = img_grey(output[i]>0.5)

        x.save('./save/video/input_%i.bmp' % (i))
        y.save('./save/video/output_%i.bmp' % (i))


def train():
    target = getSequence(np.random.randint(n))
    input = dropoutInput(target)
    
    output, hidden = model(input)

    loss = criterion(output, target)

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

    log = 100
    epochs = args.epochs

    for k in tqdm(range(1, epochs+1)):

        cost = train()

        total_cost += cost

        if k % log == 0:
            print('Iteration % d, cost: %f' % (k, total_cost/log))
            total_cost = 0
            
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
    
