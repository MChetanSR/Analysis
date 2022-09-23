import numpy as np
from .Images import rcParams
from .sigma import sigmaBlue, sigmaRed
from .fits import gaussian2DFit
import matplotlib.pyplot as plt
from scipy.constants import *

def numAtomsBlue(image, delta, s=0, plot=True, p0=None, bounds=[(), ()]):
    '''
    Calculates number of atoms from blue shadow imaging.

    Parameters:
        image: a numpy.ndarray, OD from the experiment
        delta: a float, detuning of the probe, 2*(AOMFreq-69) MHz
        imaging_params: a dictionary with keys as follows
            ex: {'binning':2, 'magnification': 2.2, 'pixelSize': 16*micro, 'saturation': 1 }
        s(optional): a float, saturation parameter of the probe. Default is 0.
        plot(optional): a bool, flag to plot the gaussian fits if True.
            Default is True.
    Returns:
        a tuple, (number of atoms from 2D gaussian fit,
        number of atoms from pixel sum, number density from gaussian fit,
        sigma_x, sigma_y, amplitude, x0, y0)
    '''
    scat = sigmaBlue(delta, 87, s)
    pOpt, pCov = gaussian2DFit(image, p0=p0, bounds=bounds, plot=plot)
    amp, xo, yo, sigma_x, sigma_y, theta, offset = pOpt
    imaging_params = rcParams().params
    pixelSize = imaging_params['pixelSize']
    magnification = imaging_params['magnification']
    binning = imaging_params['binning']
    NGaussian = 2*pi*amp*sigma_x*sigma_y*(pixelSize*binning/magnification)**2/scat
    #NIntegrate = integrate.simps(integrate.simps(image))*(pixelSize*binning/magnification)**2/scat
    NPixel = np.sum(np.sum(image, axis=0), axis=0)*(pixelSize*binning/magnification)**2/scat
    Ndensity = NGaussian/((2*pi*sigma_x*sigma_y*(pixelSize*binning/magnification)**2)**(3/2))
    return NGaussian, NPixel, Ndensity, sigma_x, sigma_y, amp, xo, yo

def numAtomsRed(image, delta, s=0, plot=True,p0=None,bounds=[(), ()]):
    """
    Calculates number of atoms from red shadow imaging.

    Parameters:
        image: a numpy.ndarray, OD from the experiment
        delta: float, detuning of the probe in kHz
        imaging_params: a dictionary with keys as follows
            ex: {'binning':2, 'magnification': 2.2, 'pixelSize': 16*micro }
        s(optional): a float, saturation parameter of the probe. Default is 0.
        plot(optional): a bool, a flag to plot the gaussian fits if True.
            Default is True.
    Returns:
        a tuple, (number of atoms from 2D gaussian fit,
        number of atoms from pixel sum, number density from gaussian fit,
        sigma_x, sigma_y, amplitude, x0, y0)
    """
    scat = sigmaRed(delta, s)
    pOpt, pCov = gaussian2DFit(image, p0=p0, bounds=bounds, plot=plot)
    amp, xo, yo, sigma_x, sigma_y, theta, offset = pOpt
    imaging_params = rcParams().params
    pixelSize = imaging_params['pixelSize']
    magnification = imaging_params['magnification']
    binning = imaging_params['binning']
    NGaussian = 2*pi*amp*sigma_x*sigma_y*(pixelSize*binning/magnification)**2/scat
    #NIntegrate = integrate.simps(integrate.simps(image))*(pixelSize*binning/magnification)**2/scat
    NPixel = np.sum(np.sum(image, axis=0), axis=0)*(pixelSize*binning/magnification)**2/scat
    Ndensity = NGaussian/((2*pi*sigma_x*sigma_y*(pixelSize*binning/magnification)**2)**(3/2))
    return NGaussian, NPixel, Ndensity, sigma_x, sigma_y, amp, xo, yo


def temperature(sizes, timeStamps, plot=True):
    '''
    Calculates temperature of the cold gas given size of the cloud at various times of flight.
    Parameters:
        sizes: 2D array, sizes of the cloud extracted from gaussian2D fit in pixel units.
            Eg: ([[x1, y1], [x2, y2], [x3, y3]])
        timeStamps: array, corresponding times of flight in s.
        plot: bool, to plot :math:`v^2` vs :math:`TOF^2`.Defaut:True.
    Returns:
        a tuple, (temperature in K along x, temperature in K along y)
    '''
    params = rcParams().params
    factor = (params['pixelSize'] * params['binning'] / params['magnification']) ** 2
    vxSqFit, intx = np.polyfit(timeStamps ** 2, factor * (sizes[:, 0] ** 2), 1)
    vySqFit, inty = np.polyfit(timeStamps ** 2, factor * (sizes[:, 1] ** 2), 1)
    tt = np.linspace(timeStamps[0], timeStamps[-1], 30)
    Tx = params['mass number'] * m_p * vxSqFit / k
    Ty = params['mass number'] * m_p * vySqFit / k
    if plot==True:
        plt.figure(figsize=(8, 3))
        plt.plot(timeStamps[:] ** 2 / (10 ** -6), sizes[:, 0] ** 2 * factor / (10 ** -12), 'o')
        plt.plot(timeStamps[:] ** 2 / (10 ** -6), sizes[:, 1] ** 2 * factor / (10 ** -12), 'o')
        plt.plot(tt ** 2 / (10 ** -6), ((vxSqFit) * (tt) ** 2 + intx) / (10 ** -12), color='C0', ls='-', label='$T_x = {:.3f}\mu K$'.format(Tx/1e-6))
        plt.plot(tt ** 2 / (10 ** -6), ((vySqFit) * (tt) ** 2 + inty) / (10 ** -12), color='C1', ls='-', label='$T_y = {:.3f}\mu K$'.format(Ty/1e-6))
        plt.xlabel('$(TOF (ms))^2$')
        plt.ylabel('$(\sigma (\mu m))^2$')
        plt.legend()
        plt.tight_layout()
    return Tx, Ty