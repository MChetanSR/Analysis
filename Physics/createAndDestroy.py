import numpy as np
from scipy.sparse import csr_matrix



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