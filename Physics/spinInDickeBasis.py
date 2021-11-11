import numpy as np
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
