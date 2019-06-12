
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


from PIL import Image

import matplotlib.pyplot as plt

def img_grey(data):
    data = data.astype(np.uint8)
    return Image.fromarray(data * 255, mode='L').convert('1')

def evaluate(weights):
    input = getSequence(0)

    model.load_state_dict(weights)

    with torch.no_grad():
        output, hidden = model(input)
    
    input = input.squeeze()
    output = output.squeeze()

    input = input.cpu().numpy()
    output = output.cpu().numpy()

    n, h, w = input.shape

    for i in range(n):
        x = img_grey(input[i]>0.5)
        y = img_grey(output[i]>0.5)

        x.save('./save/video/input_%i.bmp' % (i))
        y.save('./save/video/output_%i.bmp' % (i))


def train():
    target = getSequence(np.random.randint(n))
    input = dropotuInput(target)
    
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

    epochs = args.epochs

    for k in tqdm(range(1, epochs+1)):

        cost = train()

        total_cost += cost

        if k % 100 == 0:
            print('Iteration % d, cost: %f' % (k, total_cost/1000))
            total_cost = 0
            
            evaluate(model.state_dict())
            torch.save(model.state_dict(),  './save/weights/%d.dat' % (k))


def test():
    weight = torch.load('./save/weights/1000.dat')
    evaluate(weight)
    pass

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating moving ball')
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()
    main(args)
    
