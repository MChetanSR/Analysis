from scipy.constants import *
import math

def sigmaBlue(delta, A, s=0):
    """
    Calculates the scattering cross-section of Sr for :math:`^1S_0 \\rightarrow ^1P_1` taking into
    account the isotope shift w.r.t. A=88 isotope and hyperfine levels present
    in the excited state.

    Parameters:
        delta: float, detuning of the probe w.r.t the reference in MHz.
        s: float, saturation parameter of the probe, :math:`I/I_s`.
        A: int, mass number of the isotope.
    Returns:
        A float, Scattering cross-section.
    Comment:
        All frequencies are in MHz.
    """
    wLen = 461*nano
    Gamma = 2*pi*32
    sigma_0 = 3*wLen**2/(2*pi)
    calib = 36
    if A == 88:
        return sigma_0/(4*(2*pi*delta/Gamma)**2+1+s)
    elif A == 87:
        zeemanShift = 1.60*calib # shift in spectroscopy in choosing 87^Sr
        delta9half = delta - (zeemanShift-69)
        delta7half = delta - (zeemanShift-9.7)
        delta11half = delta - (zeemanShift-51.8)
        coeff7half = 8/(4*(2*pi*delta7half/Gamma)**2+1+s)
        coeff9half = 10/(4*(2*pi*delta9half/Gamma)**2+1+s)
        coeff11half = 12/(4*(2*pi*delta11half/Gamma)**2+1+s)
        return sigma_0*(0.1826**2)*(coeff7half+coeff9half+coeff11half)
        # the 0.1826 comes from the branching ratio for different hyperfine transitions.
    elif A == 86:
        raise ValueError('The calculation for 86^Sr is not done. Please do it.')
    elif A == 84:
        raise ValueError('The calculation for 84^Sr is not done. Please do it.')


def sigmaRed(delta, s=0):
    """
    Calculates the scattering cross-section of Sr for :math:`^1S_0 \\rightarrow ^3P_1` taking saturation
    into account.

    Parameters:
        delta: float, detuning of the probe w.r.t the reference in kHz.
        s: float, saturation parameter of the probe, :math:`I/I_s`.
    Returns:
        A float, Scattering cross-section.
    Comment:
        All frequencies are in kHz.
    """
    wLen = 689*nano
    Gamma = 7.5
    sigma_0 = 3*wLen**2/(2*pi)
    return sigma_0/(4*(delta/Gamma)**2+1+s)
