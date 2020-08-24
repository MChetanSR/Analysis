import numpy as np
from scipy import integrate
from .sigma import sigmaBlue, sigmaRed
from .fits import gaussian2DFit
from scipy.constants import *

def numAtomsBlue(image, delta, imaging_params, s=0, plot=True, isotope=87):
    '''
    Calculates number of atoms from blue shadow imaging.
    Parameters:
        image: a numpy.ndarray, OD from the experiment
        delta: a float,detuning of the probe, 2*(AOMFreq-69) MHz
        imaging_params: a dictionary with keys as follows
            ex: {'binning':2, 'magnification': 2.2, 'pixelSize': 16*micro }
        s(optional): a float, saturation parameter of the probe. Default is 0.
        plot(optional): a bool, flag to plot the gaussian fits if True.
            Default is True.
    Returns:
        a tuple: (number of atoms from 2D gaussian fit,
         number of atoms from pixel sum, sigma_x, sigma_y, amplitude, x0, y0)
    '''
    scat = sigmaBlue(delta, int(isotope), s)
    pOpt, pCov = gaussian2DFit(image, p0=None, bounds=None, plot=plot)
    amp, xo, yo, sigma_x, sigma_y, theta, offset = pOpt
    pixelSize = imaging_params['pixelSize']
    magnification = imaging_params['magnification']
    binning = imaging_params['binning']
    NGaussian = 2*pi*amp*sigma_x*sigma_y*(pixelSize*binning/magnification)**2/scat
    #NIntegrate = integrate.simps(integrate.simps(image))*(pixelSize*binning/magnification)**2/scat
    NPixel = np.sum(np.sum(image, axis=0), axis=0)*(pixelSize*binning/magnification)**2/scat
    return NGaussian, NPixel, pOpt[3], pOpt[4], pOpt[0], pOpt[1], pOpt[2]


def numAtomsRed(image, delta, imaging_params, s=0, plot=True):
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
        a tuple: (number of atoms from 2D gaussian fit,
         number of atoms from pixel sum, sigma_x, sigma_y, amplitude, x0, y0)
    """
    scat = sigmaRed(delta, s)
    pOpt, pCov = gaussian2DFit(image, p0=None, bounds=None, plot=plot)
    amp, xo, yo, sigma_x, sigma_y, theta, offset = pOpt
    pixelSize = imaging_params['pixelSize']
    magnification = imaging_params['magnification']
    binning = imaging_params['binning']
    NGaussian = 2*pi*amp*sigma_x*sigma_y*(pixelSize*binning/magnification)**2/scat
    #NIntegrate = integrate.simps(integrate.simps(image))*(pixelSize*binning/magnification)**2/scat
    NPixel = np.sum(np.sum(image, axis=0), axis=0)*(pixelSize*binning/magnification)**2/scat
    return NGaussian, NPixel, pOpt[3], pOpt[4], pOpt[0], pOpt[1], pOpt[2]

def TempShadow(OD, TOF, imaging_params):
    pass

def TempFluorescence(fl, TOF, imaging_params):
    pass