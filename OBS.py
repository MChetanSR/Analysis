import numpy as np
from scipy.integrate import solve_ivp


# ------------------------------------------
# Functions
# ------------------------------------------

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
    elif len(args) == 6:
        # for off resonant coupling
        raise NotImplementedError
    return np.transpose(trans)


# ------------------------------------------
# Classes
# ------------------------------------------

class OBSolve(object):
    def __init__(self, t, decayMatrix):
        self.t = t
        self.decayMatrix = decayMatrix

    def Hamiltonian(self, H, *HArgs):
        self.H = H
        self.HArgs = HArgs

    def Liovellian(self, t):
        H = self.H(t, *self.HArgs)
        n = len(H)
        L = np.zeros((n**2, n**2), dtype=complex)
        k = 0
        while k<n:
            for i in range(n):
                for j in range(n):
                    L[n*k+i, n*k+j] -= H[j, i]
                    L[n*i+k, n*j+k] += H[i,j]
            k+=1
        return -1j*(L) + self.decayMatrix

    def __f(self, t, rho):
        return np.dot(self.Liovellian(t), rho)

    def solve(self, initialCondition):
        t0 = self.t[0]
        tf = self.t[-1]
        rho0 = initialCondition
        sol = solve_ivp(self.__f, (t0, tf), rho0, method='RK45', t_eval=self.t)
        self.rho = sol.y
        return self.rho, sol