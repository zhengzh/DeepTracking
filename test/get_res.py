from PIL import Image
from pylab import *

import os

num = 10
ifs = ['input_%d.bmp' % (i) for i in range(0, num)]
ofs = ['output_%d.bmp' % (i) for i in range(0, num)]

d = 'save/video/'
ips = [Image.open(d+i) for i in ifs]
ops = [Image.open(d+i) for i in ofs]


# figure(figsize=(20,20), dpi=96)
# plt.rcParams['figure.figsize'] = [10, 5]

row_num = 5
fig, axes =plt.subplots(num//row_num,row_num, figsize=(10, 10), facecolor='w', edgecolor='k')

for i, ipt in enumerate(ips):
    ax = axes[i//row_num, i%row_num]
    ax.imshow(ipt)
    ax.axis('off')
    ax.set_title('input%d'%(i))

plt.savefig('inputs1.png')

fig, axes =plt.subplots(num//row_num,row_num, figsize=(10, 10), facecolor='w', edgecolor='k')

for i, ipt in enumerate(ops):
    ax = axes[i//row_num, i%row_num]
    ax.imshow(ipt)
    ax.axis('off')
    ax.set_title('outputs%d'%(i))

plt.savefig('outputs1.png')
