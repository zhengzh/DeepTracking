
from PIL import Image, ImageDraw

def draw_circle(pen, x, y, r):
    pen.ellipse((x-r, y-r, x+r, y+r), fill=(255,255, 255,255))

from random import random

w, h = 200, 200

class Ball:
    def __init__(self):
        v = 7
        self.x = random() * (w-20) + 10
        self.y = random() * (h-20) + 10
        self.vx = random() * v +1
        self.vy = random() * v + 1
        self.r = random() * 5 + 5
    
    def move(self):
        self.x += self.vx
        self.y += self.vy
        
        if self.x < self.r or self.x > w - self.r:
            self.vx = - self.vx
            
        if self.y < self.r or self.y > h - self.r:
            self.vy = - self.vy
        
    def draw(self, pen):
        draw_circle(pen, self.x, self.y, self.r)
        

balls = [Ball() for i in range(15)]


def step(balls):
    
    image = Image.new('RGB', (w, h))
    pen = ImageDraw.Draw(image)
    for ball in balls:
        ball.move()
        ball.draw(pen)
        
    return image

import matplotlib.pyplot as plt

def main(balls):
    for i in range(1000):
        image = step(balls)
        
        # plt.imshow(image)
        # plt.pause(0.01)

if __name__ == '__main__':
    main(balls)
    