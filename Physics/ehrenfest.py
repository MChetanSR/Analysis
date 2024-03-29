import numpy as np
from scipy.integrate import solve_ivp, odeint
from .basisMatrices import GellMannBasis, PauliBasis

Id, sigx, sigy, sigz = PauliBasis().matrices()

def omegaConstant(t):
    o1 = np.sqrt(1)
    if type(t)==float or type(t)==np.float64:
        return o1
    else:
        return o1*np.ones(len(t))

def omegaRamp(t, t_ramp):
    if type(t)==float or type(t)==np.float64:
        if t<=t_ramp:
            return np.sqrt(t/t_ramp)
        else:
            return 1
    else:
        conditions = [t<=t_ramp, t>t_ramp]
        values = [lambda t: np.sqrt(t/t_ramp), lambda t:1]
        return np.piecewise(t, conditions, values)

def omegaGaussian(t,amp, Tc, sigma):
    return amp*np.exp(-(t-Tc)**2/(4*sigma**2))

def omegaDoubleGaussian(t,amp, Tc1, Tc2, sigma):
    if type(t)==float or type(t)==np.float64:
        result = amp*(np.exp(-(t-Tc1)**2/(4*sigma**2))+np.exp(-(t-Tc2)**2/(4*sigma**2)))
        if result>amp:
            result=amp
    else:
        result = amp*(np.exp(-(t-Tc1)**2/(4*sigma**2))+np.exp(-(t-Tc2)**2/(4*sigma**2)))
    return result

def Map(fun, vec, args, dtype=float):
    '''
    Maps a function to a vector.

    Parameters:
        fun: function to be mapped
        vec: 1d-array, the vector to be mapped to the function
        args: tuple, secondary arguments to the function
        dtype: Default numpy.float, the data type of the elements of the array
    Returns:
        1d-array with elements of given dtype mapped by the specified function.
    '''
    result = np.zeros(len(vec), dtype=dtype)
    for i, v in enumerate(vec):
        result[i] = fun(v, *args)
    return result

class EhrenfestSU2():
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
        beta = np.arctan2(abs(omega_1), abs(omega_2))
        return alpha, beta

    def _eom(self, y, t, d1=0, d2=0, d3=0):
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
        D12 = (np.sqrt(3)/6)*(d1-d2)
        D22 = (d1+d2+4*d3)/6
        #a_s = (S11 + S22) / 2 + (D11 + D22) / 2 + (A2[0, 0] + A2[1, 1]) / 4
        b_s = (S11 - S22) / 2 + (D11 - D22) / 2 + (A2[0, 0] - A2[1, 1]) / 4
        c_s = S12 + D12 + A2[0, 1] / 2

        return [y[5]-(ax+bx*y[4]+cx*y[2]),
                y[6]-(ay+by*y[4]+cy*y[2]),
                2*(bx*y[5]+by*y[6]-b_s)*y[3],
                -2*(bx*y[5]+by*y[6]-b_s)*y[2]+2*(cx*y[5]+cy*y[6]-c_s)*y[4],
                -2*(cx*y[5]+cy*y[6]-c_s)*y[3],
                0,
                0]

    def evolve(self, t, y0, T, N=1000, p_x=0, p_y=0, d1=-1, d2=1, d3=3):
        px = np.sqrt(T)*np.random.randn(N) + p_x
        py = np.sqrt(T)*np.random.randn(N) + p_y
        #x_0 = np.sqrt(T/(30/9600)**2)*np.random.randn(N)
        #y_0 = np.sqrt(T/(60/9600)**2)*np.random.randn(N)
        self.raw = np.zeros((N, len(y0), len(t)))
        for i in range(N):
            y0[0], y0[1] = 0, 0#x_0[i], y_0[i]
            y0[5], y0[6] = px[i], py[i]
            y = odeint(self._eom, y0, t, args=(d1/2, d2/2, d3/2))
            self.raw[i] = y[:,0], y[:,1], y[:,2], y[:,3], y[:,4], y[:,5], y[:,6]
        self.result = np.mean(self.raw, axis=0)
        return self.result

    def bareStatePop(self, t):
        alpha, beta = self.mixingAngles(t)
        x, y, sx, sy, sz, px, py = self.result
        P1 = (1+sz)/2*np.sin(beta)**2+(1-sz)/2*np.cos(alpha)**2*np.cos(beta)**2+sx*np.cos(alpha)*np.cos(beta)*np.sin(beta)
        P2 = (1+sz)/2*np.cos(beta)**2+(1-sz)/2*np.cos(alpha)**2*np.sin(beta)**2-sx*np.cos(alpha)*np.cos(beta)*np.sin(beta)
        P3 = (1-sz)/2*np.sin(alpha)**2
        return np.array([P1, P2, P3])

    def bareStatePop2(self, t, result):
        alpha, beta = self.mixingAngles(t)
        x, y, sx, sy, sz, px, py = result
        P1 = (1 + sz) / 2 * np.sin(beta) ** 2 + (1 - sz) / 2 * np.cos(alpha) ** 2 * np.cos(beta) ** 2 + sx * np.cos(
            alpha) * np.cos(beta) * np.sin(beta)
        P2 = (1 + sz) / 2 * np.cos(beta) ** 2 + (1 - sz) / 2 * np.cos(alpha) ** 2 * np.sin(beta) ** 2 - sx * np.cos(
            alpha) * np.cos(beta) * np.sin(beta)
        P3 = (1 - sz) / 2 * np.sin(alpha) ** 2
        return np.array([P1, P2, P3])

    def spatialDistributions(self,t):
        alpha, beta = self.mixingAngles(t)
        x, y, sx, sy, sz, px, py = self.raw[:,]
        P1 = (1+sz)/2*np.sin(beta)**2+(1-sz)/2*np.cos(alpha)**2*np.cos(beta)**2+sx*np.cos(alpha)*np.cos(beta)*np.sin(beta)
        P2 = (1+sz)/2*np.cos(beta)**2+(1-sz)/2*np.cos(alpha)**2*np.sin(beta)**2-sx*np.cos(alpha)*np.cos(beta)*np.sin(beta)
        P3 = (1-sz)/2*np.sin(alpha)**2

class Tripod():
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
        theta = np.arccos(abs(omega_3)/omega)
        phi = np.arctan2(abs(omega_2), abs(omega_1))
        return theta, phi

def mixingAngles(omega_1, omega_2, omega_3):
    omega = np.sqrt(abs(omega_1)**2+abs(omega_2)**2+abs(omega_3)**2)
    theta = np.arccos(abs(omega_3)/omega)
    phi = np.arctan2(abs(omega_2), abs(omega_1))
    return theta, phi

def mixingAnglesSU2(omega_1, omega_2, omega_3):
    omega = np.sqrt(abs(omega_1)**2+abs(omega_2)**2+abs(omega_3)**2)
    theta = np.arccos(abs(omega_3)/omega)
    phi = np.arctan2(abs(omega_1), abs(omega_2))
    return theta, phi

def SU2_GaugeField(alpha, beta):
    A11x = (1+np.sin(beta)**2)
    A12x = 0.5*np.cos(alpha)*np.sin(2*beta)
    A22x = (np.cos(alpha)**2)*(1+np.cos(beta)**2)
    A11y = np.cos(beta)**2
    A12y = -0.5*np.cos(alpha)*np.sin(2*beta)
    A22y = (np.cos(alpha)**2)*np.sin(beta)**2
    return np.array([[A11x, A12x], [A12x, A22x]]), np.array([[A11y, A12y], [A12y, A22y]])

def SU2_ScalarTerm(alpha, beta):
    W11 = -(1+np.sin(beta)**2)
    W22 = -(np.cos(alpha)**2)*np.sin(beta)**2
    W12 = -0.5*np.cos(alpha)*np.sin(2*beta)
    return np.array([[W11, W12], [W12, W22]])

def SU3_GaugeField(theta_l, phi_l, theta_r, phi_r):
    cot_theta_l = 1/np.tan(theta_l)
    cot_theta_r = 1/np.tan(theta_r)
    alpha0 = np.sqrt(1+cot_theta_l**2+cot_theta_r**2)
    A_x = np.zeros((3, 3))
    A_y = np.zeros((3, 3))

    A_x[0, 0] = 3 + np.sin(phi_l)**2
    A_x[1, 1] = (1/alpha0**2)*(cot_theta_l**2*(4-np.sin(phi_l)**2)+2+cot_theta_r**2*np.sin(phi_r)**2)
    A_x[2, 2] = np.cos(phi_r)**2
    A_x[0, 1] = cot_theta_l * np.sin(phi_l) * np.cos(phi_l)/alpha0
    A_x[1, 2] = -cot_theta_r * np.sin(phi_r) * np.cos(phi_r) / alpha0
    A_x[1, 0] = A_x[0, 1]
    A_x[2, 1] = A_x[1, 2]

    A_y[0, 0] = np.cos(phi_l)**2
    A_y[1, 1] = (1/alpha0**2)*(cot_theta_l**2*np.sin(phi_l)**2 + cot_theta_r**2*np.sin(phi_r)**2)
    A_y[2, 2] = np.cos(phi_r)**2
    A_y[0, 1] = -cot_theta_l * np.sin(phi_l) * np.cos(phi_l) / alpha0
    A_y[1, 2] = -cot_theta_r * np.sin(phi_r) * np.cos(phi_r) / alpha0
    A_y[1, 0] = A_y[0, 1]
    A_y[2, 1] = A_y[1, 2]
    return A_x, A_y

def SU3_ScalarTerm(theta_l, phi_l, theta_r, phi_r):
    cot_theta_l = 1/np.tan(theta_l)
    cot_theta_r = 1/np.tan(theta_r)
    alpha0 = np.sqrt(1+cot_theta_l**2+cot_theta_r**2)
    Q = np.array([[3*np.sin(phi_l)**2+5, 1.5*np.sin(2*phi_l)/alpha0/np.tan(theta_l), 0],
                  [1.5*np.sin(2*phi_l)/alpha0/np.tan(theta_l),
                   ((5+3*np.cos(phi_l)**2)/np.tan(theta_l)**2+np.sin(phi_r)**2/np.tan(theta_r)**2+2)/(alpha0**2),
                   -0.5*np.sin(2*phi_r)/alpha0/np.tan(theta_r)],
                  [0, -0.5*np.sin(2*phi_r)/alpha0/np.tan(theta_r), np.cos(phi_r)**2]])
    return -Q

class EhrenfestSU3:
    def __init__(self, Tripod_l, Tripod_r):
        self.t_l = Tripod_l
        self.t_r = Tripod_r
        self.f = GellMannBasis().structureConstants()

    def _eom(self, y, t, p_x, p_y):
        theta_l, phi_l = self.t_l.mixingAngles(t)
        theta_r, phi_r = self.t_r.mixingAngles(t)
        cot_theta_l = 1/np.tan(theta_l)
        cot_theta_r = 1/np.tan(theta_r)
        alpha0 = np.sqrt(1+cot_theta_l**2+cot_theta_r**2)
        const1 = (cot_theta_l**2*(1+np.cos(phi_l)**2)-cot_theta_r**2*(1+np.cos(phi_r)**2))/(alpha0**2)
        const2 = (cot_theta_l**2*np.sin(phi_l)**2+cot_theta_r**2*np.sin(phi_r)**2)/(alpha0**2)
        e1 = (np.sin(phi_r)**2 - np.sin(phi_r)**2 + const1)/3
        e2 = (np.cos(phi_r)**2 + np.cos(phi_r) ** 2 + const2) / 3
        Ax1 = np.sin(phi_l)*np.cos(phi_l)*cot_theta_l/alpha0
        Ax2 = 0
        Ax3 = 0.5*(1+np.sin(phi_l)**2 - const1)
        Ax4 = 0
        Ax5 = 0
        Ax6 = -np.sin(phi_r)*np.cos(phi_r)*cot_theta_r/(alpha0)
        Ax7 = 0
        Ax8 = (np.sqrt(3)/2)*(2*(1+np.sin(phi_r)**2)/3+(1+np.sin(phi_l)**2)+const1)
        Ay1 = -Ax1
        Ay2 = 0
        Ay3 = 0.5*(np.cos(phi_l)**2 - const2)
        Ay4 = 0
        Ay5 = 0
        Ay6 = Ax6
        Ay7 = 0
        Ay8 = (np.sqrt(3)/2)*(-2*(np.cos(phi_r)**2)/3+const2+np.cos(phi_l)**2)
        Ax = np.array([Ax1, Ax2, Ax3, Ax4, Ax5, Ax6, Ax7, Ax8])
        Ay = np.array([Ay1, Ay2, Ay3, Ay4, Ay5, Ay6, Ay7, Ay8])
        #e = np.array([e1, e2, 0])
        equation = np.zeros(8)
        for k in range(8):
            for m in range(8):
                for j in range(8):
                    equation[k] += (p_x*Ax[m]+p_y*Ay[m])*self.f[m, k, j].real*y[j]
        return equation

    def evolve(self, t, y0, T, N=1000, p_x=0, p_y=0):
        px = np.sqrt(T)*np.random.randn(N)+p_x
        py = np.sqrt(T)*np.random.randn(N)+p_y
        self.raw = np.zeros((N, len(y0), len(t)))
        for i in range(N):
            if N==1:
                yreal = odeint(self._eom, y0, t, args=(p_x, p_y))
                y = yreal
                self.raw[i] = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], y[:, 6], y[:, 7]
            else:
                yreal = odeint(self._eom, y0, t, args=(px[i], py[i]))
                y = yreal
                self.raw[i] = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], y[:, 6], y[:, 7]
        self.result = np.mean(self.raw, axis=0)
        return self.result

    def bareStatePop(self, t):
        l = GellMannBasis().matrices()
        theta_l, phi_l = np.vectorize(self.t_l.mixingAngles)(t)
        theta_r, phi_r = np.vectorize(self.t_r.mixingAngles)(t)
        cot_theta_l = 1 / np.tan(theta_l)
        cot_theta_r = 1 / np.tan(theta_r)
        alpha0 = np.sqrt(1 + cot_theta_l**2 + cot_theta_r **2)
        z = np.zeros(len(theta_l))
        o = np.ones(len(theta_l))
        rho = np.zeros((5, 5, len(theta_l)), dtype=complex)
        D_l = np.array([np.sin(phi_l), -np.cos(phi_l), z, z, z])
        D_0 = np.array([(1 / np.tan(theta_l)) * np.cos(phi_l),
                        (1 / np.tan(theta_l)) * np.sin(phi_l), -o,
                        (1 / np.tan(theta_r)) * np.sin(phi_r),
                        (1 / np.tan(theta_r)) * np.cos(phi_r)])/alpha0
        D_r = np.array([z, z, z, -np.cos(phi_r), np.sin(phi_r)])
        D = np.array([D_l, D_0, D_r])
        Ddag = np.transpose(D, (1, 0, 2))
        for i in range(len(theta_l)):
            rho_D = l[0]/3+np.zeros((3,3), dtype=complex)
            for j in range(8):
                rho_D += self.result[j, i]*l[j+1]
            rho[:,:, i] = np.dot(np.dot(Ddag[:,:,i], rho_D), D[:, :, i])
        return np.array([abs(rho[0,0]), abs(rho[1,1]), abs(rho[2,2]), abs(rho[3,3]), abs(rho[4,4])])

    def bareStatePop2(self, t, result):
        l = GellMannBasis().matrices()
        theta_l, phi_l = np.vectorize(self.t_l.mixingAngles)(t)
        theta_r, phi_r = np.vectorize(self.t_r.mixingAngles)(t)
        cot_theta_l = 1 / np.tan(theta_l)
        cot_theta_r = 1 / np.tan(theta_r)
        alpha0 = np.sqrt(1 + cot_theta_l**2 + cot_theta_r **2)
        z = np.zeros(len(theta_l))
        o = np.ones(len(theta_l))
        rho = np.zeros((5, 5, len(theta_l)), dtype=complex)
        D_l = np.array([np.sin(phi_l), -np.cos(phi_l), z, z, z])
        D_0 = np.array([(1 / np.tan(theta_l)) * np.cos(phi_l),
                        (1 / np.tan(theta_l)) * np.sin(phi_l), -o,
                        (1 / np.tan(theta_r)) * np.sin(phi_r),
                        (1 / np.tan(theta_r)) * np.cos(phi_r)])/alpha0
        D_r = np.array([z, z, z, -np.cos(phi_r), np.sin(phi_r)])
        D = np.array([D_l, D_0, D_r])
        Ddag = np.transpose(D, (1, 0, 2))
        for i in range(len(theta_l)):
            rho_D = l[0]/3+np.zeros((3,3), dtype=complex)
            for j in range(8):
                rho_D += result[j, i]*l[j+1]
            rho[:,:, i] = np.dot(np.dot(Ddag[:,:,i], rho_D), D[:, :, i])
        return np.array([abs(rho[0,0]), abs(rho[1,1]), abs(rho[2,2]), abs(rho[3,3]), abs(rho[4,4])])