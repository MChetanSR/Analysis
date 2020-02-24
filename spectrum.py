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

def spectroscopy(ODimages, f, d=4, plot=True,fileNum='',savefig=False):
    n = len(ODimages)
    step = np.round(f[1]-f[0], 3)
    x, y = centerOfOD(ODimages[n//2])
    index = []
    for i in range(n):
        index.append(np.sum(np.sum(ODimages[i][y-d:y+d, x-d:x+d])))
    maxODAt = np.argmax(index)
    pOpt, pCov = lorentzianFit(f, np.array(index), p0=[max(index), f[maxODAt], 0.1, 0])
    if plot==True:
        plt.figure()
        plt.imshow(ODimages[maxODAt])
        plt.scatter(x, y, marker='+', color='r')
        plt.figure()
        plt.plot(f, index, 'ro')
        plt.plot(f, lorentzian(f, *pOpt), 'k', label='lor. fit')
        plt.legend()
        plt.ylabel('$\propto$ OD', fontsize=16)
        plt.xlabel('$f_{9/2 \\rightarrow 11/2}$(in MHz)', fontsize=16)
        plt.title(r'$f_{start}$ = '+str(f[0])+', $\delta f$ = '+str(step)+\
                  ', $f_0$ = '+str(np.round(pOpt[1], 3))+', file = '+fileNum)
        plt.tight_layout()
        if savefig==True:
            plt.savefig('SpectroscopyResultFor'+fileNum+'.png', transparent=True)
    return pOpt