import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import *

sigx = np.array([[0, 1],[1, 0]])
sigz = np.array([[1, 0],[0,-1]])
Id = np.eye(2)

d_unit = 0.5
t_unit = 16.5

class Ehrenfest():
    def __init__(self, omega1, omega2, omega3, omega1_args, omega2_args, omega3_args):
        self.omega1 = omega1
        self.omega1_args = omega1_args
        self.omega2 = omega2
        self.omega2_args = omega2_args
        self.omega3 = omega3
        self.omega3_args = omega3_args

    def mixingAngles(self, t):
        omega_1 = self.omega1(t, *self.omega1_args)
        omega_2 = self.omega2(t, *self.omega2_args)
        omega_3 = self.omega3(t, *self.omega3_args)
        omega = np.sqrt(abs(omega_1)**2+abs(omega_2)**2+abs(omega_3)**2)
        alpha = np.arccos(abs(omega_3)/omega)
        beta = np.arctan(abs(omega_1/omega_2))
        return alpha, beta
    '''
    def Hamiltonian(self, t, p=0, d1=0, d2=0, d3=0):
        A = np.eye(2)
        W = np.eye(2)
        D = np.eye(2)
        H = (p**2/2)*np.eye(2) - np.dot(p*np.eye(2), A) + np.matmul(A, A)/2 + W + D
        return H
    '''
    def _eom(self, t, y, p_x=0, p_y=0, d1=0, d2=0, d3=0):
        alpha, beta = self.mixingAngles(t)
        A11x = (1+np.sin(beta)**2)
        A12x = 0.5*np.cos(alpha)*np.sin(2*beta)
        A22x = (np.cos(alpha)**2)*(1+np.cos(beta)**2)
        A11y = np.cos(beta)**2
        A12y = -0.5*np.cos(alpha)*np.sin(2*beta)
        A22y = (np.cos(alpha)**2)*np.sin(beta)**2
        ax = (A11x + A22x) / 2
        bx = (A11x - A22x) / 2
        cx = A12x
        ay = (A11y + A22y) / 2
        by = (A11y - A22y) / 2
        cy = A12y
        Ax = (ax * Id + bx * sigz + cx * sigx)
        Ay = (ay * Id + by * sigz + cy * sigx)
        A2 = np.matmul(Ax, Ax)+np.matmul(Ay, Ay)
        S11 = (np.sin(beta)**2+1) - A2[0, 0]/2
        S22 = np.cos(alpha)**2 * (1 + np.cos(beta)**2) - A2[1, 1]/2
        S12 = 0.5*np.cos(alpha) * np.sin(2*beta) - A2[0, 1]/2
        D11 = (d1+d2)/2
        D12 = np.sqrt(3)/6*(d1-d2)
        D22 = (d1+d2+4*d3)/6
        #a_s = (S11 + S22) / 2 + (D11 + D22) / 2 + (A2(1, 1) + A2(2, 2)) / 4
        b_s = (S11 - S22) / 2 + (D11 - D22) / 2 + (A2[0, 0] - A2[1, 1]) / 4
        c_s = S12 + D12 + A2[0, 1] / 2

        return [y[5]-(ax+bx*y[4]+cx*y[2]),
                y[6]-(ay+by*y[4]+cy*y[2]),
                2*(bx*y[5]+by*y[6]-b_s)*y[3],
                -2*(bx*y[5]+by*y[6]-b_s)*y[2]+2*(cx*y[5]+cy*y[6]-c_s)*y[4],
                -2*(cx*y[5]+cy*y[6]-c_s)*y[3],
                p_x,
                p_y]

    def evolve(self, t, y0, T, N=1000, p_x=0, p_y=0, d1=-1, d2=1, d3=3):
        d1 *= d_unit
        d2 *= d_unit
        d3 *= d_unit
        px = np.sqrt(T)*np.random.randn(N) + p_x
        py = np.sqrt(T)*np.random.randn(N) + p_y
        result = np.zeros((N, len(y0), len(t)))
        for i in range(N):
            result[i] = solve_ivp(self._eom,(t[0], t[-1]), y0, args=(px[i], py[i], d1, d2, d3), t_eval=t)['y'].reshape((len(y0), len(t)))
        self.result = np.mean(result, axis=0)
        return self.result

    def bareStatePop(self, t):
        alpha, beta = self.mixingAngles(t)
        x, y, sx, sy, sz, px, py = self.result
        P1 = (1+sz)/2*np.sin(beta)**2+(1-sz)/2*np.cos(alpha)**2*np.cos(beta)**2+sx*np.cos(alpha)*np.cos(beta)*np.sin(beta)
        P2 = (1+sz)/2*np.cos(beta)**2+(1-sz)/2*np.cos(alpha)**2*np.sin(beta)**2-sx*np.cos(alpha)*np.cos(beta)*np.sin(beta)
        P3 = (1-sz)/2*np.sin(alpha)**2
        return P1, P2, P3
