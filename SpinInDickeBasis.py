import numpy as np
#import matplotlib.pyplot as plt
#import scipy.linalg as ln
from scipy.sparse import csr_matrix
#import sympy as sy


def S_p(s):
    if s<0:
        raise ValueError('spin has to be non-negative')
    else:
        m_s = np.arange(-s, s+1, 1)
        rows = np.arange(0,2*s,1)
        cols = np.arange(1,2*s+1,1)
        values = np.array([np.sqrt(s*(s+1)-m*(m+1)) for m in m_s[:-1]])
        spar = csr_matrix((values, (rows, cols)), shape=(int(2*s+1), int(2*s+1)))
        return spar

def S_m(s):
    if s<0:
        raise ValueError('spin has to be non-negative')
    else:
        m_s = np.arange(-s, s+1, 1)
        rows = np.arange(1,2*s+1,1)
        cols = np.arange(0,2*s,1)
        values = np.array([np.sqrt(s*(s+1)-m*(m-1)) for m in m_s[1:]])
        spar = csr_matrix((values, (rows, cols)), shape=(int(2*s+1), int(2*s+1)))
        return spar

def S_x(s):
    return (S_p(s)+S_m(s))/2

def S_y(s):
    return (S_p(s)-S_m(s))/2j

def S_z(s):
    if s<0:
        raise ValueError('spin has to be non-negative')
    else:
        m_s = np.arange(-s, s+1, 1)
        rows = np.arange(0,2*s+1,1)
        cols = np.arange(0,2*s+1,1)
        values = np.array([m for m in m_s[:]])
        spar = csr_matrix((values, (rows, cols)), shape=(int(2*s+1), int(2*s+1)))
        return spar
def a(n):
    if n<0:
        raise ValueError('number of bosons must be greater than 0!')