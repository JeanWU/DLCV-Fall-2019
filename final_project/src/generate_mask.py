import random
import numpy as np
import sys
import glob
import os
from PIL import Image

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def random_walk(canvas, ini_x, ini_y, length):
    x = ini_x
    y = ini_y
    x_size, y_size = canvas.shape
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=x_size - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=y_size - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 0
    return canvas


img_folder = sys.argv[1]
save_train = sys.argv[2]
save_gt = sys.argv[3]

img_list = []
for path in glob.glob(os.path.join(img_folder, '*.jpg')):
    img_list.append(path)

img_list.sort()
for idx, path in enumerate(img_list):
    img = Image.open(path)
    h, w = img.size

    canvas = np.ones((w, h)).astype("i")
    ini_x = random.randint(0, h - 1)
    ini_y = random.randint(0, w - 1)
    mask = random_walk(canvas, ini_x, ini_y, h*w)
    img = np.array(img)
    new_mask = np.stack((mask,)*3, axis=-1)
    mask_x = img * new_mask

    mask = Image.fromarray(mask * 255).convert('1')
    mask.save(os.path.join(save_train, '{:03d}_mask.jpg'.format(idx+501)))
    mask_x = Image.fromarray(mask_x.astype('uint8'))
    mask_x = Image.eval(mask_x, (lambda x: 255 if x == 0 else x))
    mask_x.save(os.path.join(save_train, '{:03d}_masked.jpg'.format(idx+501)))
    img = Image.fromarray(img.astype('uint8'))
    img.save(os.path.join(save_gt, '{:03d}.jpg'.format(idx+501)))

