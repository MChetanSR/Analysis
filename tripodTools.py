from math import sin, cos
import pandas as pd

def detunings(p, theta):
    d_rec = 0.004796
    sig_plus = 80
    sig_minus = 80.024
    pi_f = 80.012
    d_1, d_2, d_3 = (-1-p*cos(theta)*2)*d_rec, (1-p*sin(theta)*2)*d_rec, (3+p*cos(theta)*2)*d_rec
    return pd.DataFrame([[sig_plus-d_1, sig_minus-d_3, pi_f-d_2],[-d_1, -d_3, -d_2]],
                 ['Freq.', '$\delta_{ij}$'], ['$\sigma_+$', '$\sigma_-$', '$\pi$'])