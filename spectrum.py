from .fits import lorentzian, lorentzianFit
import matplotlib.pyplot as plt
import matplotlib.patches as patch
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
    try:
        pOpt, pCov = lorentzianFit(f, np.array(index), p0=[max(index), f[maxODAt], 0.1, 0])
    except RuntimeError:
        pOpt = None

    if plot == True:
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,8))
        ax[0].imshow(ODimages[maxODAt])
        ax[0].scatter(x, y, marker='+', color='r')
        rectangle = patch.Rectangle((x-d, y-d), 2*d, 2*d, linewidth=1,edgecolor='r',facecolor='none')
        ax[0].add_patch(rectangle)
        ax[1].plot(f, index, 'ro')
        if pOpt!=None:
            ax[1].plot(f, lorentzian(f, *pOpt), 'k', label=r'lor. fit \n $\Gamma=$'+str(np.round(pOpt[2], 3))+
                                                         ', $f_0$ = '+str(np.round(pOpt[1], 3)))
            ax[1].legend()
        ax[1].set_ylabel('$\propto$ OD', fontsize=16)
        ax[1].set_xlabel('$f_{9/2 \\rightarrow 11/2}$(in MHz)', fontsize=16)
        ax[1].set_title(r'$f_{start}$ = '+str(f[0])+', $f_{step}$ = '+str(step)+', file = '+fileNum)
        plt.tight_layout()
        if savefig==True:
            plt.savefig('SpectroscopyResultFor'+fileNum+'.png', transparent=True)
    return pOpt # amp, centre, gamma, offset