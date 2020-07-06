from .fits import lorentzian, lorentzianFit, gaussian2DFit, bv, bvFit
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from scipy.constants import *
import numpy as np

def spectroscopy(ODimages, f, d=4, plot=True, fileNum='', savefig=False):
    '''
    Adds the OD of the pixels around the centre and uses the sum to plot the spectrum of the scan
    corresponding to the given frequencies
    Args:
        ODimages: ODimages extracted from ShadowImaging sequences
        f: array of frequencies for which the scan is done
        d: int, to specify size of the image area to consider around the centre of OD
        plot: bool, default is True to specify if the data has to be plotted
        fileNum: string, the image file number for which the analysis is done
        savefig: bool, default is False. Change it to true if you want to save the spectrum as .png

    Returns:
        pOpt = (amp, centre, gamma, offset) of the lorentzian fit
    '''
    n = len(ODimages)
    if n!=len(f):
        raise ValueError('No of images and no. of  frequencies are not equal')
    step = np.round(f[1]-f[0], 3)
    x, y = np.shape(ODimages)[1:]//2
    index = []
    for i in range(n):
        index.append(np.sum(np.sum(ODimages[i][y-d:y+d, x-d:x+d])))
    maxODAt = np.argmax(index)
    try:
        pOpt, pCov = lorentzianFit(f, np.array(index), p0=[max(index), f[maxODAt], 0.1, 0])
    except RuntimeError:
        pOpt = []

    if plot == True:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
        i = ax[0].imshow(ODimages[maxODAt])
        fig.colorbar(i, ax=ax[0])
        ax[0].scatter(x, y, marker='+', color='r')
        rectangle = patch.Rectangle((x-d, y-d), 2*d, 2*d, linewidth=1,edgecolor='r',facecolor='none')
        ax[0].add_patch(rectangle)
        ax[0].grid(False)
        ax[1].plot(f, index, 'ro')
        if pOpt!=[]:
            ax[1].plot(f, lorentzian(f, *pOpt), 'k', label=r'lor. fit: $\Gamma=$'+str(np.round(pOpt[2], 3))+
                                                         ', $f_0$ = '+str(np.round(pOpt[1], 3)))
            ax[1].legend()
        ax[1].set_ylabel('\propto OD', fontsize=16)
        ax[1].set_xlabel('$f$(in MHz)', fontsize=16)
        ax[1].set_title(r'$f_{start}$ = '+str(f[0])+', $f_{step}$ = '+str(step)+', file = '+fileNum)
        plt.tight_layout()
        if savefig==True:
            plt.savefig('SpectroscopyResultFor'+fileNum+'.png', transparent=True)
    return pOpt # amp, centre, gamma, offset


def spectroscopyFaddeva(ODimages, f, plot=True, fileNum='', savefig=False):
    '''
    Fits od images to a gaussian and uses amp*sigma_x*sigma_y to plot the spectrum of the scan
    corresponding to the given frequencies. This is fit to b_v(\delta) from Chang Chi's thesis.
    Args:
        ODimages: ODimages extracted from ShadowImaging sequences
        f: array of frequencies for which the scan is done
        plot: bool, default is True to specify if the data has to be plotted
        fileNum: string, the image file number for which the analysis is done
        savefig: bool, default is False. Change it to true if you want to save the spectrum as .png

    Returns:
        pOpt = (amp, centre, gamma, offset) of the lorentzian fit
    '''
    n = len(ODimages)
    if n!=len(f):
        raise ValueError('No of images and no. of  frequencies are not equal')
    step = np.round(f[1]-f[0], 3)
    x, y = np.shape(ODimages)[1:]//2
    index = []
    for i in range(n):
        p0 = [0.5, x, y, x/2, y/2, 0, 0.1]
        bounds = ([0, x-4, y-4, x/5, y/5, 0, -0.1], [1.5, x+4, y+4, 4*x/5, 4*y/5, 6.28, 0.3])
        amp, xo, yo, sx, sy, theta, offset = gaussian2DFit(ODimages[i], p0, bounds)
        index.append(amp*sx*sy)
    maxODAt = np.argmax(index)
    try:
        pOpt, pCov = bvFit(f, np.array(index), p0=[max(index), f[maxODAt], 0.1, 0])
        T = (pOpt[2]**2)*(87*m_p)/k
    except RuntimeError:
        pOpt = []

    if plot == True:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
        ax.grid(False)
        ax.plot(f, index, 'ro')
        if pOpt!=[]:
            ax.plot(f, bv(f, *pOpt), 'k', label=r'fit: T='+str(np.round(T, 2))+
                                                  '$b_0(0)$='+str(np.round(pOpt[1], 1))+
                                                    ', $f_0$ = '+str(np.round(pOpt[0], 3)))
            ax.legend()
        ax.set_ylabel('$OD \\times \sigma_x \\times \sigma_y$', fontsize=16)
        ax.set_xlabel('$f$(in MHz)', fontsize=16)
        ax.set_title(r'$f_{start}$ = '+str(f[0])+', $f_{step}$ = '+str(step)+', file = '+fileNum)
        plt.tight_layout()
        if savefig==True:
            plt.savefig('SpectroscopyResultFor'+fileNum+'.png', transparent=True)
    return pOpt # amp, centre, gamma, offset