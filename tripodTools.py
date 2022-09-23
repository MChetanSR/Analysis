from math import sin, cos
import numpy as np
import pandas as pd


def detunings(p, theta):
    d_rec = 0.004796
    sig_plus = 80
    sig_minus = 80.024
    pi_f = 80.012
    d_1, d_2, d_3 = (-1-p*cos(theta)*2)*d_rec, (1-p*sin(theta)*2)*d_rec, (3+p*cos(theta)*2)*d_rec
    return pd.DataFrame([[sig_plus-d_1, sig_minus-d_3, pi_f-d_2],[-d_1, -d_3, -d_2]],
                 ['Freq.', '$\delta_{ij}$'], ['$\sigma_+$', '$\sigma_-$', '$\pi$'])

def detunings2(p_x, p_y):
    d_rec = 0.004796
    sig_plus = 80
    sig_minus = 80.024
    pi_f = 80.012
    d_1, d_2, d_3 = (-1-p_x*2)*d_rec, (1-p_y*2)*d_rec, (3+p_x*2)*d_rec
    return pd.DataFrame([[sig_plus-d_1, sig_minus-d_3, pi_f-d_2],[-d_1, -d_3, -d_2]],
                 ['Freq.', '$\delta_{ij}$'], ['$\sigma_+$', '$\sigma_-$', '$\pi$'])

def BStoDS(*args):
    '''
    Change of basis matrix from bare state basis {0, 1, 2, 3} to dressed basis {D1, D2, B1, B2}
    '''
    if len(args) == 3:
        O1, O2, O3 = list(map(np.conjugate, args))
        O = np.sqrt(abs(O1)**2+abs(O2)**2+abs(O3)**2)
        phi = np.arctan2(abs(O1), abs(O2))
        eps = abs(O3)/abs(O)
        ph1 = O1*np.conjugate(O3)/(abs(O1)*abs(O3))
        ph2 = O2*np.conjugate(O3)/(abs(O2)*abs(O3))
        trans = np.array([[0, np.cos(phi)*ph1, -np.sin(phi)*ph2, 0],
                          [0, eps*np.sin(phi)*ph1, eps*np.cos(phi)*ph2, -np.sqrt(1-eps**2)],
                          [1/np.sqrt(2), 1/np.sqrt(2)*O1/O, 1/np.sqrt(2)*O2/O, 1/np.sqrt(2)*O3/O],
                          [-1/np.sqrt(2), 1/np.sqrt(2)*O1/O, 1/np.sqrt(2)*O2/O, 1/np.sqrt(2)*O3/O]])
    else:
        raise(NotImplementedError)
    return np.transpose(trans)

def Ds(p1, p2, p3, p4, p5):
    '''
    Given the bare state populations of five +ve :math:`m_F` ground states, calculates the state in bark state basis.

    .. math:: |\psi \\rangle = d_l e^{i \\alpha} |D_l \\rangle + d_0|D_0 \\rangle+ d_r e^{i \\beta}|D_r \\rangle


    Parameters:
        p1: population of :math:`|m_F = 1/2 \\rangle`
        p2: population of :math:`|m_F = 3/2 \\rangle`
        p3: population of :math:`|m_F = 5/2 \\rangle`
        p4: population of :math:`|m_F = 7/2 \\rangle`
        p5: population of :math:`|m_F = 9/2 \\rangle`
    Returns:
        a tuple of :math:`d_l`, :math:`d_0`, :math:`d_r`, :math:`\\alpha`, :math:`\\beta` and
        state array :math:`|\psi \\rangle`
    '''
    d0 = np.sqrt(abs(2*p3))
    dl = np.sqrt(abs(p1+p2-p3/2))
    dr = np.sqrt(abs(p4+p5-p3/2))
    alpha = np.arctan2(np.sqrt(abs(4*p3*(p1+p2-p3/2-(p1-p2)**2))), p1-p2)
    beta = np.arctan2(np.sqrt(abs(4*p3*(p4+p5-p3/2-(p5-p4)**2))), p5-p4)
    state = [dl*np.exp(1j*alpha), d0, dr*np.exp(1j*beta)]
    return dl, d0, dr, alpha, beta, state


