
from PIL import Image, ImageDraw

def draw_circle(pen, x, y, r):
    pen.ellipse((x-r, y-r, x+r, y+r), fill=(255,255, 255,255))

from random import random

# w, h = 25, 25
w, h = 50, 50
# w, h = 100, 100

class Ball:
    def __init__(self):
        v = 2
        self.x = random() * (w-20) + 10
        self.y = random() * (h-20) + 10
        self.vx = random() * v
        self.vy = random() * v
        #self.r = random() * 3 + 1
        self.r = 1
    
    def move(self):
        self.x += self.vx
        self.y += self.vy
        
        if self.x < self.r or self.x > w - self.r:
            self.vx = - self.vx
            
        if self.y < self.r or self.y > h - self.r:
            self.vy = - self.vy
        
    def draw(self, pen):
        draw_circle(pen, self.x, self.y, self.r)
        

balls = [Ball() for i in range(10)]


def step(balls):
    
    image = Image.new('RGB', (w, h))
    pen = ImageDraw.Draw(image)
    for ball in balls:
        ball.move()
        ball.draw(pen)
    
    res = np.array(image)[:,:,0]>100
    return res

import matplotlib.pyplot as plt

def main(balls):
    pass

def test(balls):
    for i in range(1000):
        image = step(balls)
        plt.imshow(image)
        plt.pause(0.01)

import numpy as np
from tqdm import tqdm
def generate_data(args):
    
    n = args.num
    l = args.seq_len
    
    data = np.zeros([n, l , w, h], dtype=np.uint8)

    for num in tqdm(range(n)):
        balls = [Ball() for _ in range(5)]

        for t in range(l):
            image = step(balls)
            # plt.imshow(image)
            # plt.pause(0.1)
            data[num, t, :, :] = image
    
    np.save('./save/data2', data)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating moving ball')
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('-w', type=int, default=100)
    
    args = parser.parse_args()
    w = h = args.w
    generate_data(args)

    # test(balls)
