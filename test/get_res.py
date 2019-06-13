from PIL import Image
from pylab import *

import os

num = 50
ifs = ['input_%d.bmp' % (i) for i in range(0, num)]
ofs = ['output_%d.bmp' % (i) for i in range(0, num)]

d = 'save/video/'
ips = [Image.open(d+i) for i in ifs]
ops = [Image.open(d+i) for i in ofs]


# figure(figsize=(20,20), dpi=96)
# plt.rcParams['figure.figsize'] = [10, 5]

fig, axes =plt.subplots(10,10, figsize=(20, 20), facecolor='w', edgecolor='k')

for i, ipt in enumerate(ips):
    ax = axes[i//10, i%10]
    ax.imshow(ipt)
    ax.axis('off')
    ax.set_title('input%d'%(i))

plt.savefig('inputs1.png')

fig, axes =plt.subplots(10,10, figsize=(20, 20), facecolor='w', edgecolor='k')

for i, ipt in enumerate(ops):
    ax = axes[i//10, i%10]
    ax.imshow(ipt)
    ax.axis('off')
    ax.set_title('outputs%d'%(i))

plt.savefig('outputs1.png')
