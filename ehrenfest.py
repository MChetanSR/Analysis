import numpy as np
from scipy.integrate import solve_ivp, odeint
from SpinInDickeBasis import SU3Basis

sigx = np.array([[0, 1],[1, 0]])
sigz = np.array([[1, 0],[0,-1]])
Id = np.eye(2)

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

class EhrenfestSU3:
    def __init__(self, Tripod_l, Tripod_r):
        self.t_l = Tripod_l
        self.t_r = Tripod_r
        self.f = SU3Basis.structureConstants()

    def _eom(self, y, t):
        theta_l, phi_l = self.t_l.mixingAngles(t)
        theta_l, phi_r = self.t_r.mixingAngles(t)
        cot_theta_l = 1/np.tan(theta_l)
        cot_theta_r = 1/np.tan(theta_r)
        alpha0 = sqrt(1+cot_theta_l ^ 2+cot_theta_r ^ 2)
        e1 = (cot_theta_l**2*(1+np.cos(phi_l)**2)-cot_theta_r**22*(1+np.cos(phi_r)**2))/(3*alpha0**2)
        e2 = (cot_theta_l**2*np.sin(phi_l)**2+cot_theta_r**2*np.sin(phi_r)**2)/(3*alpha0**2)
        Ax1 = np.sin(phi_l)*np.cos(phi_l)*cot_theta_l/(2*alpha0)
        Ax2 = 0
        Ax3 = 0.5*(1+np.sin(phi_l)**2-(cot_theta_l**2*(1+np.cos(phi_l)**2)-cot_theta_r**2*(1+cos(phi_r)**2)))/(alpha0**2)
        Ax4 = 0
        Ax5 = 0
        Ax6 = -np.sin(phi_r)*np.cos(phi_r)*cot_theta_r/(2*alpha0)
        Ax7 = 0
        Ax8 = (sqrt(3)/2)*(1+np.sin(phi_r)**2+e1)
        Ay1 = -Ax1
        Ay2 = 0
        Ay3 = 0.5*(np.cos(phi_l)**2-(cot_theta_l**2*np.sin(phi_l)**2+cot_theta_r**2*np.sin(phi_r)**2))/(alpha0**2)
        Ay4 = 0
        Ay5 = 0
        Ay6 = Ax6
        Ay7 = 0
        Ay8 = (np.sqrt(3)/2)*(-np.cos(phi_r)**2+e2)
        Ax = np.array([Ax1, Ax2, Ax3, Ax4, Ax5, Ax6, Ax7, Ax8])
        Ay = np.array([Ay1, Ay2, Ay3, Ay4, Ay5, Ay6, Ay7, Ay8])
        e = np.array([e1, e2, 0])

        equation = np.zeros(8)
        for k in range(8):
            for m in range(8):
                for j in range(8):
                    equation[k] += (p_x*Ax[m]+p_y*Ay[m])*self.f(m, k, j)*y(j)
        return equation

    def evolve(self, t, y0, T, N=1000, p_x=0, p_y=0):
        px = np.sqrt(T)*np.random.randn(N)+p_x
        py = np.sqrt(T)*np.random.randn(N)+p_y
        # x_0 = np.sqrt(T/(30/9600)**2)*np.random.randn(N)
        # y_0 = np.sqrt(T/(60/9600)**2)*np.random.randn(N)
        self.raw = np.zeros((N, len(y0), len(t)))
        for i in range(N):
            y0[8], y0[9] = px[i], py[i]
            y = odeint(self._eom, y0, t, args=(d1/2, d2/2, d3/2))
            self.raw[i] = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], y[:, 6]
        self.result = np.mean(self.raw, axis=0)
        return self.result

    def bareStatePop(self):
        pass
