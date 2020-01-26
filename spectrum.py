from .fits import lorentzian, lorentzianFit
import matplotlib.pyplot as plt
import numpy as np


def centerOfOD(image):
    image = image + 3
    sy, sx = image.shape
    m = np.sum(np.sum(image, axis=0), axis=0)
    x, y = 0, 0
    for i in range(0, sy):
        y += i * np.sum(image[i]) / m
    for i in range(0, sx):
        x += i * np.sum(image[:, i]) / m
    return int(x), int(y)

def spectrograph(ODimages, fStart, fStep, d=4, plot=True):
    n = len(ODimages)
    f = fStart + fStep * np.arange(n)
    x, y = centerOfOD(ODimages[n//2])
    index = []
    for i in range(n):
        index.append(np.sum(np.sum(ODimages[i][y-d:y+d, x-d:x+d])))
    pOpt, pCov = lorentzianFit(f, np.array(index), p0=[max(index), f[np.argmax(index)], 0.1, 0])
    if plot==True:
        plt.figure()
        plt.imshow(ODimages[n // 2])
        plt.scatter(x, y, marker='+', color='r')
        plt.figure()
        plt.plot(f, index, 'ro')
        plt.plot(f, lorentzian(f, *pOpt), 'k', label='lor. fit')
        plt.legend()
        plt.title(r'$f_{start}$ = '+str(fStart)+', $\delta f$ = '+str(fStep)+', $f_0$ = '+str(np.round(pOpt[1], 3)))
    return pOpt