import cv2
import numpy as np
import os
from hough import houghLine
import matplotlib.pyplot as plt
import math
from math import *

path='/Users/mridulsahi/101903128_MRIDUL_SAHI_CV_ ASSIGNMENT/img01.jpg'
image=cv2.imread(path,0)
'''
image = np.zeros((150,150))
image[10, 10] = 1
image[20, 20] = 1
image[30, 30] = 1
'''
accumulator, thetas, rhos = houghLine(image)
plt.figure('Original Image')
plt.imshow(image)
plt.set_cmap('gray')
plt.figure('Hough Space')
plt.imshow(accumulator)
plt.set_cmap('gray')
plt.show()
idx = np.argmax(accumulator)
rho = int(rhos[int(idx / accumulator.shape[1])])
theta = thetas[int(idx % accumulator.shape[1])]
m=-cos(theta)/sin(theta)
c=rho/math.sin(theta)
print("m=")
print(m)
print("c=")
print(c)