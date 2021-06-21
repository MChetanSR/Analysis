from scipy.constants import * # all physical constants
import numpy as np # numpy for array manipulations
# look for sympy documentation for the arguments
from sympy.physics.wigner import wigner_6j, wigner_3j
from sympy.physics.wigner import clebsch_gordan as CG

# function definitions and classes
au = physical_constants['atomic unit of electric polarizability'][0]

class ket():
    '''
    Definition of a state with appropriate quantum numbers as shown in the __init__.
    '''
    def __init__(self, n, L, S, J, I, F, mF):
        # usual definition of quantum numbers
        self.n = n
        self.L = L
        self.S = S
        self.J = J
        self.I = I
        self.F = F
        self.m = mF

    def is_ket(self):
        return True

    def is_bra(self):
        return False

    def __rmul__(self, other):
        if other.is_bra():
            if (other.n == self.n) and \
               (other.L == self.L) and \
               (other.S == self.S) and \
               (other.J == self.J) and \
               (other.I == self.I) and \
               (other.F == self.F) and \
               (other.m == self.m):
                return 1
            else:
                return 0
        else:
            raise TypeError('Multiplication defined only with a bra!')

    def __str__(self):
        # string representation for easy viewing
        return '|n={0}, L={1}, S={2}, J={3}, I={4}, F={5}, mF={6}>'\
                .format(self.n, self.L, self.S, self.J, self.I, self.F, self.m)

class bra(ket):
    def __init__(self, n, L, S, J, I, F, mF):
        super().__init__(n, L, S, J, I, F, mF)

    def is_ket(self):
        return False

    def is_bra(self):
        return True

    def __str__(self):
        # string representation for easy viewing
        return '<n={0}, L={1}, S={2}, J={3}, I={4}, F={5}, mF={6}|'\
                .format(self.n, self.L, self.S, self.J, self.I, self.F, self.m)


def alphaTensor(F, K, eps):
    '''
    Full polarizability tensor eq. 5 in PRA reference
    
    Parameters:
    -----------
    F: Total angular momentum of the required ground state
    K: coefficients in irreducible representation
    eps: polarization vector of the light (E_x, E_y, E_z)
    '''
    alpha = np.zeros((int(2*F+1), int(2*F+1)), dtype=complex)
    for x in range(int(2*F+1)):
        for y in range(int(2*F+1)):
            for j in range(3):
                for M in range(-j, j+1, 1):
                    alpha[x, y] += K[j]*((-1)**M)*CG(F, j, F, y-F, -M, x-F)*tens(j, M, *eps)
    return alpha

def alpha(F, mF, K, eps):
    '''
    Polarizability of each mF.
    
    Parameters:
    -----------
    F: Total angular momentum of the required ground state.
    mF: magnetic quantum number.
    K: coefficients in irreducible representation
    eps: polarization vector of the light (E_x, E_y, E_z)
    '''
    alpha = 0
    for j in range(3):
        for M in range(-j, j+1, 1):
                    alpha += float(K[j]*((-1)**M)*CG(F, j, F, -mF, -M, -mF)*tens(j, M, *eps))
    return alpha

def tens(j, M, ux, uy, uz):
    '''
    Helper function to convert polarization of the light into compound tensor
    components using equation 12 in epjd reference of cesium
    '''
    u = [(ux-1j*uy)/np.sqrt(2), uz, -(ux+1j*uy)/np.sqrt(2)]
    s = 0
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            s += (-1)**(M+y)*np.conjugate(u[x+1])*u[-y+1]*np.sqrt(2*j+1)*wigner_3j(1, j, 1, x, -M, y)
    return s
            

def B_fs(k_i, k_f):
    '''
    Branching ratio for fine structure.
    
    Parameters:
    -----------
    k_i: excited state
    k_f: ground state
    '''
    return (2*k_i.J+1)*(2*k_f.L+1)*(wigner_6j(k_f.J, 1, k_i.J, k_i.L, k_i.S, k_f.L))**2

def B_hfs(k_i, k_f):
    '''
    Branching ratio of hyperfine structure
    
    Parameters:
    -----------
    k_i: excited state
    k_f: ground state
    '''
    return (2*k_i.F+1)*(2*k_f.J+1)*(wigner_6j(k_f.F, 1, k_i.F, k_i.J, k_i.I, k_f.J))**2

def K(j, k_f, k_is, w_l, params):
    '''
    Coefficients in irreducible representation of polarizability as shown in
    equation 5 of PRA reference.
    
    Parameters:
    -----------
    j: 0 for scalar, 1 for vector, 2 for tensor coefficient
    k_f: ground state
    k_is: list of all excited states
    w_l: wavelength
    params: spectroscopic parameters of all excited states freq., z costant, decay rate
    '''
    w_0s, zs, rates = params[:,0]*1e15, params[:,1], params[:,2]*1e6
    summation = 0
    for i, k_i in enumerate(k_is):
        constant = (3*pi*epsilon_0*c**3/(w_0s[i]**3))*zs[i]*rates[i]
        d = abs(float((2*k_i.F+1)*B_hfs(k_f, k_i)*B_fs(k_f, k_i)*constant))
        k_j = ((-1)**(k_i.F+k_f.F-j))*np.sqrt((2*j+1)/(2*k_f.F+1))*\
               (wigner_6j(k_f.F, k_f.F, j, 1, 1, k_i.F))*d
        summation += float(k_j)*(1/(w_0s[i] + w_l) + ((-1)**j)/(w_0s[i]-w_l))
    return float(summation)