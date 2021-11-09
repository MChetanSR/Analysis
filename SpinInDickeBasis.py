import numpy as np
#import matplotlib.pyplot as plt
#import scipy.linalg as ln
from scipy.sparse import csr_matrix, kron
from scipy.sparse import identity as Id

def S_m(s):
    '''
    Matrix representation of ladder lowering operator for spin, in general angular momentum.

    Parameters:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of :math:`S^-` of shape (2s+1, 2s+1) in Dicke basis, {:math:`|s, m_s\\rangle`}.
    '''
    if s<0:
        raise ValueError('spin has to be non-negative')
    else:
        m_s = -np.arange(-s, s+1, 1)
        rows = np.arange(0,2*s,1)
        cols = np.arange(1,2*s+1,1)
        values = np.array([np.sqrt(s*(s+1)-m*(m-1)) for m in m_s[:-1]])
        spar = csr_matrix((values, (rows, cols)), shape=(int(2*s+1), int(2*s+1)))
        return spar

def S_p(s):
    '''
    Matrix representation of ladder raising operator for spin, in general angular momentum.

    Parameters:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of :math:`S^+` of shape (2s+1, 2s+1) in Dicke basis, {:math:`|s, m_s\\rangle`}.
    '''
    if s<0:
        raise ValueError('spin has to be non-negative')
    else:
        m_s = -np.arange(-s, s+1, 1)
        rows = np.arange(1,2*s+1,1)
        cols = np.arange(0,2*s,1)
        values = np.array([np.sqrt(s*(s+1)-m*(m+1)) for m in m_s[1:]])
        spar = csr_matrix((values, (rows, cols)), shape=(int(2*s+1), int(2*s+1)))
        return spar

def S_x(s):
    '''
    Matrix representation of x component of the operator for spin, in general angular momentum.

    Parameters:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of :math:`S_x` of shape (2s+1, 2s+1) in Dicke basis, {:math:`|s, m_s\\rangle`}.
    '''
    return (S_p(s)+S_m(s))/2

def S_y(s):
    '''
    Matrix representation of y component of the operator for spin, in general angular momentum.

    Parameters:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of :math:`S_y` of shape (2s+1, 2s+1) in Dicke basis, {:math:`|s, m_s\\rangle`}.
    '''
    return (S_p(s)-S_m(s))/2j

def S_z(s):
    '''
    Matrix representation of z component of the operator for spin, in general angular momentum.

    Parameters:
        s: spin quantum number.
    Returns:
        a sparse matrix representation of :math:`S_z` of shape (2s+1, 2s+1) in Dicke basis, {:math:`|s, m_s\\rangle`}.
    '''
    if s<0:
        raise ValueError('spin has to be non-negative')
    else:
        m_s = -np.arange(-s, s+1, 1)
        rows = np.arange(0,2*s+1,1)
        cols = np.arange(0,2*s+1,1)
        values = np.array([m for m in m_s[:]])
        spar = csr_matrix((values, (rows, cols)), shape=(int(2*s+1), int(2*s+1)))
        return spar

def Spin(s):
    '''
    Matrix representation of quantum mechanical spin, in general angular momentum, s.

    Parameters:
        s: int or half int, the spin quantum number.
    Returns:
        a tuple of sparse matrices corresponding to :math:`S_x, S_y, S_z`
    '''
    return S_x(s), S_y(s), S_z(s)

def SpinAngularMomenta(I, L, S):
    '''
    Returns angular momenta operators of a state given I, L, S in the tensor product basis.

    Parameters:
        I: nuclear spin quantum number of the atomic state
        L: orbital angular momentum quantum number
        S: spin quantum number of the state
    Returns:
        a tuple of angular momenta, :math:`((I_x, I_y, I_z), (L_x, L_y, L_z), (S_x, S_y, S_z))`
        (each a tuple of components as sparse matrices) in tensor product space.

    '''
    I_x = kron(kron(S_x(I), Id(2 * L + 1)), Id(2 * S + 1))
    I_y = kron(kron(S_y(I), Id(2 * L + 1)), Id(2 * S + 1))
    I_z = kron(kron(S_z(I), Id(2 * L + 1)), Id(2 * S + 1))
    L_x = kron(kron(Id(2 * I + 1), S_x(L)), Id(2 * S + 1))
    L_y = kron(kron(Id(2 * I + 1), S_y(L)), Id(2 * S + 1))
    L_z = kron(kron(Id(2 * I + 1), S_z(L)), Id(2 * S + 1))
    s_x = kron(kron(Id(2 * I + 1), Id(2 * L + 1)), S_x(S))
    s_y = kron(kron(Id(2 * I + 1), Id(2 * L + 1)), S_y(S))
    s_z = kron(kron(Id(2 * I + 1), Id(2 * L + 1)), S_z(S))
    IOperator = (I_x, I_y, I_z)
    LOperator = (L_x, L_y, L_z)
    SOperator = (s_x, s_y, s_z)
    return IOperator, LOperator, SOperator

def a(N=5):
    '''
    Matrix representation of bosonic destruction operator.

    Parameters:
        n: destroy boson at n in an N state Fock space
        N: dimensions of the Fock space, default 5
    Returns:
        a sparse matrix representation of a of shape (N, N) in Fock basis, {:math:`|n_1, n_2, n_3, ..., N\\rangle`}.
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

    Parameters:
        n: create a boson at n in an N state Fock space
        N: dimensions of the Fock space, default 5
    Returns:
        a sparse matrix representation of a of shape (N, N) in Fock basis, {:math:`|n_1, n_2, n_3, ..., N\\rangle`}.
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

class SU3Basis():
    def _init__(self):
        return self

    def matrices(self):
        I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        l1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        l2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
        l3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        l4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        l5 = np.array([[0, 0,-1j], [0, 0, 0], [1j, 0, 0]])
        l6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        l7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        l8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])/np.sqrt(3)
        return np.array([I, l1, l2, l3, l4, l5, l6, l7, l8])

    def structureConstants(self):
        f = np.zeros((8, 8, 8), dtype=complex)
        l = self.matrices()
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    f[i,j,k] = -1j*np.trace(np.dot(np.dot(l[i+1], l[j+1]), l[k+1]) - np.dot(np.dot(l[j+1], l[i+1]), l[k+1]))/4
        self.f = f
        return self.f

    def structureConstant(self, i, j, k):
        return self.f[i-1, j-1, k-1]