import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import *
import matplotlib.pyplot as plt

def gaussian(x, amplitude, xo, sigma, offset):
    g = offset + amplitude*np.exp(-(x-xo)**2/(2*sigma**2))
    return g


def gaussianFit(array, p0=None, bounds=None, plot=True):
    """
    Fits the given array to an 1D-gaussian.
    Parameters:
        array: 1darray, the data to fit to the gaussian
        p0: ndarray, initial guess for the fit params in the form of
            [amplitude, xo, sigma, offset]. Default is None.
        bounds: tuple of lower bound and upper bound for the fit.
            Default is None.
    Returns:
        pOpt: optimized parameters in the same order as p0
        pCov: covarience parameters of the fit.
        Read scipy.optimize.curve_fit for details.
    """
    x = np.arange(0,len(array))
    pOpt, pCov = curve_fit(gaussian, x, array, p0, bounds)
    if plot==True:
        pass
    return pOpt,pCov


def gaussian2D(X, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x = X[0]
    y = X[1]
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = np.sin(2*theta)*(-1/(2*sigma_x**2) + 1/(2*sigma_y**2))
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x-xo)**2) + b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


def gaussian2DFit(image, p0=None, bounds=None, plot=True):
    """
    Fits an image with a 2D gaussian.
    Parameters:
        image: numpy ndarray
        p0: ndarray, initial guess for the fit params in the form of
            [amplitude, xo, yo, sigma_x, sigma_y, theta, offset].
            Default None (fits for OD images).
        bounds: tuple of lower bound and upper bound for the fit.
            Default None (fits for OD images)
        plot:  bool to show the plot of the fit. Default True.
    Returns:
        pOpt: optimized parameters in the same order as p0
        pCov: covarience parameters of the fit.
        Read scipy.optimize.curve_fit for details.
    """
    Ny, Nx = image.shape
    x = np.linspace(0, Nx, Nx, endpoint=False)
    y = np.linspace(0, Ny, Ny, endpoint=False)
    X = np.meshgrid(x, y)
    if (p0==None and bounds==None):
        p0 = [0.5, Nx/2, Ny/2, Nx/4, Ny/4, 0, 0]
        b = ([-0.5, 0.25*Nx, 0.25*Ny, 0.1*Nx, 0.1*Ny, -0.1, -0.5],\
             [10, 0.75*Nx, 0.75*Ny, 0.7*Nx, 0.7*Ny, 0.1, 1])
    pOpt, pCov = curve_fit(gaussian2D, X, image.reshape((Nx*Ny)), p0, bounds=b)
    fit = gaussian2D(X, *pOpt).reshape(Ny, Nx)
    if plot==True:
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
        ax[0].plot(image[int(pOpt[2])], 'r.')
        ax[0].plot(x, gaussian2D(np.meshgrid(x,int(pOpt[2])), *pOpt), 'k')
        ax[0].set_xlabel('x (pixels)')
        ax[0].set_ylabel('Optical depth')
        ax[0].set_ylim(pOpt[6]-0.2, pOpt[6]+pOpt[0]+0.5)
        ax[1].plot(image[:, int(pOpt[1])], 'r.')
        ax[1].plot(y, gaussian2D(np.meshgrid(int(pOpt[1]),y), *pOpt), 'k')
        ax[1].set_xlabel('y (pixels)')
        ax[1].set_ylim(pOpt[6]-0.2, pOpt[6]+pOpt[0]+0.5)
        ax[2].contour(X[0], X[1], fit, cmap=plt.cm.hot, vmin=-0.1, vmax=pOpt[0]+0.5)
        matrix=ax[2].imshow(image, cmap=plt.cm.hot, vmin=-0.1, vmax=pOpt[0]+0.5)
        ax[2].set_xlabel('x (pixels)')
        ax[2].set_ylabel('y (pixels)')
        f.colorbar(matrix)
        plt.tight_layout()
    return pOpt, pCov

def lorentzian(x, amplitude, xo, gamma, offset):
    l = offset + amplitude/((x-xo)**2+(gamma/2)**2)
    return l


def lorentzianFit(array, p0=None, bounds=None, plot=True):
    """
    Fits the given array to a Lorentzian.
    Parameters:
        array: 1darray, the data to fit to the lorentzian
        p0: ndarray, initial guess for the fit params in the form of
            [amplitude, xo, gamma, offset]. Default is None.
        bounds: tuple of lower bound and upper bound for the fit.
            Default is None.
    Returns:
        pOpt: optimized parameters in the same order as p0
        pCov: covarience parameters of the fit.
        Read scipy.optimize.curve_fit for details.
    """
    x = np.arange(0, len(array))
    pOpt, pCov = curve_fit(lorentzian, x, array, p0, bounds)
    if plot==True:
        pass
    return pOpt,pCov