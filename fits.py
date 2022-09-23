import numpy as np
from scipy.optimize import curve_fit
from scipy.constants import *
import matplotlib.pyplot as plt
import matplotlib as mpl

def gaussian(x, amplitude, xo, sigma_x, offset):
    g = offset + amplitude*np.exp(-(x-xo)**2/(2*sigma_x**2))
    return g

def gaussianFit(x, array, p0=[], bounds=[(), ()], plot=True):
    """
    Fits the given array to an 1D-gaussian.

    Parameters:
        x: 1darray, the argument values of the gaussian
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

def gaussian2DFit(image, p0=None, bounds=[(), ()], plot=True, title=''):
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
    if (p0==None and bounds==[(),()]):
        p0 = [0.5, Nx/2, Ny/2, Nx/4, Ny/4, 0, 0]
        bounds = ([-0.5, 0.4*Nx, 0.4*Ny, 0.0*Nx, 0.0*Ny, -0.1, -0.5],\
             [10, 0.6*Nx, 0.6*Ny, 0.7*Nx, 0.7*Ny, 0.1, 1])
    else:
        p0 = p0
        bounds = bounds
    pOpt, pCov = curve_fit(gaussian2D, X, image.reshape((Nx*Ny)), p0, bounds=bounds)
    fit = gaussian2D(X, *pOpt).reshape(Ny, Nx)
    if plot==True:
        f, ax = plt.subplots(nrows=1, ncols=5, gridspec_kw={'width_ratios': [4,4,4,4,0.2]}, figsize=(16, 4))
        if int(pOpt[2])>=len(y):
            ax[0].plot(image[-1], 'r.')
        elif int(pOpt[2])<0:
            ax[0].plot(image[0], 'r.')
        else:
            ax[0].plot(image[int(pOpt[2]), :], 'r.')
        ax[0].plot(x, gaussian2D(np.meshgrid(x,int(pOpt[2])), *pOpt), 'k')
        ax[0].set_xlabel('x (pixels)')
        if int(pOpt[1])>=len(x):
            ax[1].plot(image[:, -1], 'r.')
        elif int(pOpt[1])<0:
            ax[1].plot(image[:, 0], 'r.')
        else:
            ax[1].plot(image[:, int(pOpt[1])], 'r.')
        ax[1].plot(y, gaussian2D(np.meshgrid(int(pOpt[1]),y), *pOpt), 'k')
        ax[1].set_xlabel('y (pixels)')
        if pOpt[0]<0.2:
            ax[0].set_ylim(pOpt[6]-pOpt[0]*1.2, pOpt[6]+pOpt[0]*1.2)
            ax[1].set_ylim(pOpt[6]-pOpt[0]*1.2, pOpt[6]+pOpt[0]*1.2)
        ax[2].contour(X[0], X[1], fit, cmap=plt.cm.hot, vmin=pOpt[6]-pOpt[0]*0.12, vmax=pOpt[-1]+abs(pOpt[0])*1.2)
        matrix=ax[2].imshow(image, cmap=plt.cm.hot, vmin=pOpt[6]-pOpt[0]*0.12, vmax=pOpt[-1]+abs(pOpt[0])*1.2)
        ax[2].set_xlabel('x (pixels)')
        ax[2].set_ylabel('y (pixels)')
        ax[2].grid(False)
        ax[3].imshow(image-fit, cmap=plt.cm.hot, vmin=pOpt[6]-pOpt[0]*0.12, vmax=pOpt[-1]+abs(pOpt[0])*1.2)
        ax[3].set_xlabel('x (pixels)')
        ax[3].set_ylabel('y (pixels)')
        ax[3].grid(False)
        f.colorbar(matrix, cax=ax[4])
        f.suptitle(str(title))
        plt.tight_layout()
    return pOpt, pCov


def threeGaussian2D(X, *args):
    amplitudes, xos, yos, sigma_xs, sigma_ys, thetas, offsets = np.array(args).reshape((7, 3))
    x = X[0]
    y = X[1]
    g = 0
    for i in range(len(amplitudes)):
        g += gaussian2D(X, amplitudes[i], xos[i], yos[i], sigma_xs[i], sigma_ys[i], thetas[i], offsets[i])
    return g.ravel()


def threeGaussian2DFit(image, p0, bounds, TOF, plot=True, cropSize=6, logNorm=False):
    Ny, Nx = image.shape
    x = np.linspace(0, Nx, Nx, endpoint=False)
    y = np.linspace(0, Ny, Ny, endpoint=False)
    X = np.meshgrid(x, y)
    # first Gaussian or thick Gaussian should be well fit first. p0 and bounds should take care of this.
    croppedImage = image[p0[2]-cropSize:p0[2]+cropSize, p0[1]-cropSize:p0[1]+cropSize]
    Xc = np.meshgrid(np.arange(0, 2*cropSize), np.arange(0, 2*cropSize))
    croppedP0 = (p0[0], cropSize, cropSize, p0[3], p0[4], p0[5], p0[6])
    b = bounds
    b[0][1], b[0][2] = 0, 0
    b[1][1], b[1][2] = 3*cropSize/2, 3*cropSize/2
    pOpt, pCov = gaussian2DFit(croppedImage, p0=croppedP0,bounds=b, plot=False)
    pOpt[1] = p0[1]-cropSize+pOpt[1]
    pOpt[2] = p0[2]-cropSize+pOpt[2]
    disQuanta = hbar*(2*pi/(689*nano*87*m_p))*TOF*milli
    dis = disQuanta*2.5/(16*micro)
    p0 = [[pOpt[0], pOpt[0]*0.4, pOpt[0]*0.3],
          [pOpt[1], pOpt[1]-dis, pOpt[1]],
          [pOpt[2], pOpt[2]-dis, pOpt[2]-2*dis],
          [pOpt[3], pOpt[3], pOpt[3]],
          [pOpt[4], pOpt[4], pOpt[4]],
          [pOpt[5], pOpt[5], pOpt[5]],
          [pOpt[6], pOpt[6], pOpt[6]]]
    bounds = ([[pOpt[0]*0.0, pOpt[0]*0.0, pOpt[0]*0.0],
               [pOpt[1]-0.1, pOpt[1]-dis*1.3, pOpt[1]*0.7],
               [pOpt[2]-0.1, pOpt[2]-dis*1.3, pOpt[2]-2*dis*1.3],
               [pOpt[3]*0.8, pOpt[3]*0.8, pOpt[3]*0.8],
               [pOpt[4]*0.8, pOpt[4]*0.8, pOpt[4]*0.8],
               [pOpt[5]-0.1, pOpt[5]-0.1, pOpt[5]-0.1],
               [pOpt[6]-150, pOpt[6]-150, pOpt[6]-150]],
              [[pOpt[0]*1.5, pOpt[0]*1.5, pOpt[0]*1.5],
               [pOpt[1]+0.1, pOpt[1]-dis*0.7, pOpt[1]*1.3],
               [pOpt[2]+0.1, pOpt[2]-dis*0.7, pOpt[2]-2*dis*0.7],
               [pOpt[3]*1.2, pOpt[3]*1.2, pOpt[3]*1.2],
               [pOpt[4]*1.2, pOpt[4]*1.2, pOpt[4]*1.2],
               [pOpt[5]+0.2, pOpt[5]+0.2, pOpt[5]+0.2],
               [pOpt[6]+150, pOpt[6]+150, pOpt[6]+150]])
    pOpt, pCov = curve_fit(threeGaussian2D, X, image.reshape((Nx*Ny)), p0=np.array(p0).reshape((21)), bounds=np.array(bounds).reshape((2, 21)))
    fit = threeGaussian2D(X, *pOpt).reshape(Ny, Nx)
    if plot == True:
        f, ax = plt.subplots(nrows=1, ncols=4, gridspec_kw={'width_ratios': [4, 0.2, 4, 0.2]}, figsize=(10, 4))
        if logNorm==True:
            ax[0].contour(X[0], X[1], fit, cmap=plt.cm.hot, norm=mpl.colors.LogNorm(vmin=1e-1, vmax=1e-1+abs(pOpt[0])*1.2))
            matrix = ax[0].imshow(image, aspect='auto', cmap=plt.cm.hot, norm=mpl.colors.LogNorm(vmin=1e-1, vmax=9e-1+abs(pOpt[0])*1.2))
            residue = ax[2].imshow(image-fit, aspect='auto', cmap=plt.cm.hot)
            ax[2].set_title('Residue:'+str(np.round(np.sum(np.sum(image-fit)), 5)))
        else:
            ax[0].contour(X[0], X[1], fit, cmap=plt.cm.hot, vmin=pOpt[-1], vmax=pOpt[-1]+abs(pOpt[0])*1.2)
            matrix = ax[0].imshow(image, aspect='auto', cmap=plt.cm.hot, vmin=pOpt[-1], vmax=pOpt[-1]+abs(pOpt[0])*1.2)
            residue = ax[2].imshow(image-fit, aspect='auto', cmap=plt.cm.hot)
        ax[0].set_xlabel('x (pixels)')
        ax[0].set_ylabel('y (pixels)')
        ax[0].grid(False)
        ax[2].set_xlabel('x(pixels)')
        ax[2].set_ylabel('y(pixels)')
        ax[2].grid(False)
        f.colorbar(matrix, cax=ax[1])
        f.colorbar(residue, cax=ax[3])
        plt.tight_layout()
    return pOpt, pCov


def multipleGaussian2D(X, *args):
    amplitudes, xos, yos, sigma_xs, sigma_ys, thetas, offsets = np.array(args).reshape((7,5))
    x = X[0]
    y = X[1]
    g = 0
    for i in range(len(amplitudes)):
        g += gaussian2D(X, amplitudes[i], xos[i], yos[i], sigma_xs[i], sigma_ys[i], thetas[i], offsets[i])
    return g.ravel()

def multipleGaussian2DFit(image, p0, bounds, TOF, plot=True, cropSize=6, tolerence=0.3, logNorm=False):
    Ny, Nx = image.shape
    x = np.linspace(0, Nx, Nx, endpoint=False)
    y = np.linspace(0, Ny, Ny, endpoint=False)
    X = np.meshgrid(x, y)
    # first Gaussian or thick Gaussian should be well fit first. p0 and bounds should take care of this.
    croppedImage = image[int(p0[2])-cropSize:int(p0[2])+cropSize, int(p0[1])-cropSize:int(p0[1])+cropSize]
    Xc = np.meshgrid(np.arange(0, 2*cropSize), np.arange(0, 2*cropSize))
    croppedP0 = (p0[0], cropSize, cropSize, p0[3], p0[4], p0[5], p0[6])
    b = bounds
    b[0][1], b[0][2] = 0, 0
    b[1][1], b[1][2] = 3*cropSize/2, 3*cropSize/2
    pOpt, pCov = gaussian2DFit(croppedImage, p0=croppedP0,bounds=b, plot=False)
    pOpt[1] = p0[1]-cropSize+pOpt[1]
    pOpt[2] = p0[2]-cropSize+pOpt[2]
    disQuanta = hbar*(2*pi/(689*nano*87*m_p))*TOF*milli
    dis = disQuanta*2.5/(16*micro)
    d = dis
    '''
    p0 = [[pOpt[0], pOpt[0]*0.4, pOpt[0]*0.3, pOpt[0]*0.02, pOpt[0]*0.02, pOpt[0]*0.02, pOpt[0]*0.02, pOpt[0]*0.02],
          [pOpt[1], pOpt[1]-d, pOpt[1], pOpt[1]-d, pOpt[1], pOpt[1]+d, pOpt[1]+d, pOpt[1]-d],
          [pOpt[2], pOpt[2]-d, pOpt[2]-2*d, pOpt[2]-3*d, pOpt[2]-4*d, pOpt[2]-d, pOpt[2]+d, pOpt[2]+d],
          [pOpt[3], pOpt[3], pOpt[3], pOpt[3], pOpt[3], pOpt[3], pOpt[3], pOpt[3]],
          [pOpt[4], pOpt[4], pOpt[4], pOpt[4], pOpt[4], pOpt[4], pOpt[4], pOpt[4]],
          [pOpt[5], pOpt[5], pOpt[5], pOpt[5], pOpt[5], pOpt[5], pOpt[5], pOpt[5]],
          [pOpt[6], pOpt[6], pOpt[6], pOpt[6], pOpt[6], pOpt[6], pOpt[6], pOpt[6]]]
    x = tolerence
    ld = (1-x)*d
    ud = (1+x)*d
    bounds = ([[-pOpt[0]*0.1, -pOpt[0]*0.1, -pOpt[0]*0.1, -pOpt[0]*0.1, -pOpt[0]*0.1, -pOpt[0]*0.1, -pOpt[0]*0.1, -pOpt[0]*0.1],
               [pOpt[1]-x*d, pOpt[1]-ud, pOpt[1]-x*d, pOpt[1]-ud, pOpt[1]-x*d, pOpt[1]+ld, pOpt[1]+ld, pOpt[1]-ud],
               [pOpt[2]-x*d, pOpt[2]-ud, pOpt[2]-2*ud, pOpt[2]-3*ud, pOpt[2]-4*ud, pOpt[2]-ud, pOpt[2]+ld, pOpt[2]+ld],
               [pOpt[3]*0.8, pOpt[3]*0.8, pOpt[3]*0.8, pOpt[3]*0.8, pOpt[3]*0.8, pOpt[3]*0.8, pOpt[3]*0.8, pOpt[3]*0.8],
               [pOpt[4]*0.8, pOpt[4]*0.8, pOpt[4]*0.8, pOpt[4]*0.8, pOpt[4]*0.8, pOpt[4]*0.8, pOpt[4]*0.8, pOpt[4]*0.8],
               [pOpt[5]-0.1, pOpt[5]-0.1, pOpt[5]-0.1, pOpt[5]-0.1, pOpt[5]-0.1, pOpt[5]-0.1, pOpt[5]-0.1, pOpt[5]-0.1],
               [pOpt[6]-150, pOpt[6]-150, pOpt[6]-150, pOpt[6]-150, pOpt[6]-150, pOpt[6]-150, pOpt[6]-150, pOpt[6]-150]],
              [[pOpt[0]*1.5, pOpt[0]*1.5, pOpt[0]*1.5, pOpt[0]*0.7, pOpt[0]*0.7, pOpt[0]*0.5, pOpt[0]*0.5, pOpt[0]*0.5],
               [pOpt[1]+x*d, pOpt[1]-ld, pOpt[1]+x*d, pOpt[1]-ld, pOpt[1]+x*d, pOpt[1]+ud, pOpt[1]+ud, pOpt[1]-ld],
               [pOpt[2]+x*d, pOpt[2]-ld, pOpt[2]-2*ld, pOpt[2]-3*ld, pOpt[2]-4*ld, pOpt[2]-ld, pOpt[2]+ud, pOpt[2]+ud],
               [pOpt[3]*1.2, pOpt[3]*1.2, pOpt[3]*1.2, pOpt[3]*1.2, pOpt[3]*1.2, pOpt[3]*1.2, pOpt[3]*1.2, pOpt[3]*1.2],
               [pOpt[4]*1.2, pOpt[4]*1.2, pOpt[4]*1.2, pOpt[4]*1.2, pOpt[4]*1.2, pOpt[4]*1.2, pOpt[4]*1.2, pOpt[4]*1.2],
               [pOpt[5]+0.2, pOpt[5]+0.2, pOpt[5]+0.2, pOpt[5]+0.2, pOpt[5]+0.2, pOpt[5]+0.2, pOpt[5]+0.2, pOpt[5]+0.2],
               [pOpt[6]+150, pOpt[6]+150, pOpt[6]+150, pOpt[6]+150, pOpt[6]+150, pOpt[6]+150, pOpt[6]+150, pOpt[6]+150]])
    '''
    p0 = [[pOpt[0], pOpt[0] * 0.4, pOpt[0] * 0.3, pOpt[0] * 0.02, pOpt[0] * 0.02],
          [pOpt[1], pOpt[1] - dis, pOpt[1], pOpt[1] - dis, pOpt[1]],
          [pOpt[2], pOpt[2] - dis, pOpt[2] - 2 * dis, pOpt[2] - 3 * dis, pOpt[2] - 4 * dis],
          [pOpt[3], pOpt[3], pOpt[3], pOpt[3], pOpt[3]],
          [pOpt[4], pOpt[4], pOpt[4], pOpt[4], pOpt[4]],
          [pOpt[5], pOpt[5], pOpt[5], pOpt[5], pOpt[5]],
          [pOpt[6], pOpt[6], pOpt[6], pOpt[6], pOpt[6]]]
    bounds = (
    [[pOpt[0] * -0.1, pOpt[0] * -0.1, pOpt[0] * -0.1, pOpt[0] * -0.1, pOpt[0] * -0.1],
     [pOpt[1] - 0.1*dis, pOpt[1] - dis * 1.1, pOpt[1] - 0.1 * dis, pOpt[1] - dis * 1.1, pOpt[1] - 0.1*dis],
     [pOpt[2] - 0.1*dis, pOpt[2] - dis * 1.1, pOpt[2] - 2 * dis * 1.1, pOpt[2] - 3 * dis * 1.1, pOpt[2] - 4 * dis * 1.1],
     [pOpt[3] * 0.9, pOpt[3] * 0.9, pOpt[3] * 0.9, pOpt[3] * 0.9, pOpt[3] * 0.9],
     [pOpt[4] * 0.9, pOpt[4] * 0.9, pOpt[4] * 0.9, pOpt[4] * 0.9, pOpt[4] * 0.9],
     [pOpt[5] - 0.1, pOpt[5] - 0.1, pOpt[5] - 0.1, pOpt[5] - 0.1, pOpt[5] - 0.1],
     [pOpt[6] - 150, pOpt[6] - 150, pOpt[6] - 150, pOpt[6] - 150, pOpt[6] - 150]],
    [[pOpt[0] * 1.5, pOpt[0] * 1.5, pOpt[0] * 1.5, pOpt[0] * 0.7, pOpt[0] * 0.7],
     [pOpt[1] + 0.1*dis, pOpt[1] - dis * 0.9, pOpt[1] * 1.1, pOpt[1] - dis * 0.9, pOpt[1] + 0.1*dis],
     [pOpt[2] + 0.1*dis, pOpt[2] - dis * 0.9, pOpt[2] - 2 * dis * 0.9, pOpt[2] - 3 * dis * 0.9, pOpt[2] - 4 * dis * 0.9],
     [pOpt[3] * 1.1, pOpt[3] * 1.1, pOpt[3] * 1.1, pOpt[3] * 1.1, pOpt[3] * 1.1],
     [pOpt[4] * 1.1, pOpt[4] * 1.1, pOpt[4] * 1.1, pOpt[4] * 1.1, pOpt[4] * 1.1],
     [pOpt[5] + 0.2, pOpt[5] + 0.2, pOpt[5] + 0.2, pOpt[5] + 0.2, pOpt[5] + 0.2],
     [pOpt[6] + 150, pOpt[6] + 150, pOpt[6] + 150, pOpt[6] + 150, pOpt[6] + 150]])#'''
    pOpt, pCov = curve_fit(multipleGaussian2D, X, image.reshape((Nx*Ny)), p0=np.array(p0).reshape((35)), bounds=np.array(bounds).reshape((2, 35)))
    fit = multipleGaussian2D(X, *pOpt).reshape(Ny, Nx)
    if plot == True:
        f, ax = plt.subplots(nrows=1, ncols=4, gridspec_kw={'width_ratios': [4, 0.2, 4, 0.2]}, figsize=(10, 4))
        if logNorm==True:
            ax[0].contour(X[0], X[1], fit, cmap=plt.cm.hot, norm=mpl.colors.LogNorm(vmin=1e-1, vmax=1e-1+abs(pOpt[0])*1.2))
            matrix = ax[0].imshow(image, aspect='auto', cmap=plt.cm.hot, norm=mpl.colors.LogNorm(vmin=1e-1, vmax=9e-1+abs(pOpt[0])*1.2))
            residue = ax[2].imshow(image-fit, aspect='auto', cmap=plt.cm.hot)
            ax[2].set_title('Residue:'+str(np.round(np.sum(np.sum(image-fit)), 5)))
        else:
            ax[0].contour(X[0], X[1], fit, cmap=plt.cm.hot, vmin=pOpt[-1], vmax=pOpt[-1]+abs(pOpt[0])*1.2) 
            matrix = ax[0].imshow(image, aspect='auto', cmap=plt.cm.hot, vmin=pOpt[-1], vmax=pOpt[-1]+abs(pOpt[0])*1.2)
            residue = ax[2].imshow(image-fit, aspect='auto', cmap=plt.cm.hot)
        ax[0].set_xlabel('x (pixels)')
        ax[0].set_ylabel('y (pixels)')
        ax[0].grid(False)
        ax[2].set_xlabel('x(pixels)')
        ax[2].set_ylabel('y(pixels)')
        ax[2].grid(False)
        f.colorbar(matrix, cax=ax[1])
        f.colorbar(residue, cax=ax[3])
        plt.tight_layout()
    return pOpt, pCov

def lorentzian(x, amplitude, xo, gamma, offset):
    l = offset + amplitude*(gamma/2)/((x-xo)**2+(gamma/2)**2)
    return l

def lorentzianFit(x, array, p0=[], bounds=[(), ()], plot=False):
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
    pOpt, pCov = curve_fit(lorentzian, x, array, p0, bounds=bounds)
    if plot==True:
        raise NotImplementedError
    return pOpt,pCov

