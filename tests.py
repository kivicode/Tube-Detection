import numpy as np
import cv2


def load(i):
    return cv2.imread(f'images/{i}.png')


print(load("img"))
