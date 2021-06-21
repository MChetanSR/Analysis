import numpy as np
#import matplotlib.pyplot as plt
#import scipy.linalg as ln
from scipy.sparse import csr_matrix
#import sympy as sy


def S_p(s):
    '''
    Matrix representation of ladder raising operator for spin.
    Args:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of s+ of shape (2s+1, 2s+1) in Dicke basis, {|s, m_s>}.
    '''
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
    '''
    Matrix representation of ladder lowering operator for spin.
    Args:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of s- of shape (2s+1, 2s+1) in Dicke basis, {|s, m_s>}.
    '''
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
    '''
    Matrix representation of x component of the operator for spin.
    Args:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of s_x of shape (2s+1, 2s+1) in Dicke basis, {|s, m_s>}.
    '''
    return (S_p(s)+S_m(s))/2

def S_y(s):
    '''
    Matrix representation of y component of the operator for spin.
    Args:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of s_y of shape (2s+1, 2s+1) in Dicke basis, {|s, m_s>}.
    '''
    return (S_p(s)-S_m(s))/2j

def S_z(s):
    '''
    Matrix representation of z component of the operator for spin.
    Args:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of s_z of shape (2s+1, 2s+1) in Dicke basis, {|s, m_s>}.
    '''
    if s<0:
        raise ValueError('spin has to be non-negative')
    else:
        m_s = np.arange(-s, s+1, 1)
        rows = np.arange(0,2*s+1,1)
        cols = np.arange(0,2*s+1,1)
        values = np.array([m for m in m_s[:]])
        spar = csr_matrix((values, (rows, cols)), shape=(int(2*s+1), int(2*s+1)))
        return spar

def a(N=5):
    '''
    Matrix representation of bosonic destruction operator.
    Args:
        n: destroy boson at n in an N state Fock space
        N: dimensions of the Fock space, default 5
    Returns:
        a sparse matrix representation of a of shape (N, N) in Fock basis, {|n_1, n_2, n_3, ..., N>}.
    '''
    if N<1:
        raise ValueError('number of bosons must be greater than 1!')
    else:
        n_n = np.arange(0, N, 1)
        rows = np.arange(0, N-1, 1)
        cols = np.arange(1, N, 1)
        values = np.array([np.sqrt(n) for n in n_n[1:]])
        spar = csr_matrix((values, (rows, cols)), shape=(int(N), int(N)))
        return spar

def a_dag(N=5):
    '''
    Matrix representation of bosonic creation operator.
    Args:
        n: create a boson at n in an N state Fock space
        N: dimensions of the Fock space, default 5
    Returns:
        a sparse matrix representation of a of shape (N, N) in Fock basis, {|n_1, n_2, n_3, ..., N>}.
    '''
    if N<1:
        raise ValueError('number of bosons must be greater than 0 and less than 1!')
    else:
        n_n = np.arange(0, N, 1)
        rows = np.arange(1, N, 1)
        cols = np.arange(0, N-1, 1)
        values = np.array([np.sqrt(n+1) for n in n_n[:-1]])
        spar = csr_matrix((values, (rows, cols)), shape=(int(N), int(N)))
        return spar

class bosonic(object):
    def __init__(self, N):
        self.N = N

    def operators(self):
        N = self.N
        n_n = np.arange(0, N, 1)
        rows = np.arange(0, N-1, 1)
        cols = np.arange(1, N, 1)
        values = np.array([np.sqrt(n) for n in n_n[1:]])
        spar1 = csr_matrix((values, (rows, cols)), shape=(int(N), int(N)))
        rows = np.arange(1, N, 1)
        cols = np.arange(0, N-1, 1)
        values = np.array([np.sqrt(n+1) for n in n_n[:-1]])
        spar2 = csr_matrix((values, (rows, cols)), shape=(int(N), int(N)))
        return spar1, spar2
