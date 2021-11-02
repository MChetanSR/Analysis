#!python
#cython: language_level=3
#cython: language = c++


from libc.math cimport sqrt, cos, sin, acos, atan, abs
cimport numpy as np
import numpy as np
import cython
from scipy.integrate import odeint


cdef np.ndarray sigx = np.array([[0, 1], [1, 0]], dtype=np.float)
cdef np.ndarray sigz = np.array([[1, 0], [0, -1]], dtype=np.float)
cdef np.ndarray Id = np.array([[1, 0], [0, 1]], dtype=np.float)

cpdef double omegaConstant(double t):
    o1 = np.sqrt(1)
    if type(t) == float or type(t) == np.float64:
        return o1
    else:
        return o1 * np.ones(len(t))

def omegaRamp(double t, double t_ramp):
    if type(t) == float or type(t) == np.float64:
        if t <= t_ramp:
            return np.sqrt(t / t_ramp)
        else:
            return 1
    else:
        conditions = [t <= t_ramp, t > t_ramp]
        values = [lambda t: np.sqrt(t / t_ramp), lambda t: 1]
        return np.piecewise(t, conditions, values)

def omegaGaussian(double t, double amp, double Tc, double sigma):
    '''
    A function to output gaussian pulse in time with given center and sigma.
    Parameters: 
        t: double, time at which the value on gaussian is needed
        amp: double, amplitude of the gaussian
        Tc: double, center of the gaussian
        sigma: double, sigma of the gaussian
    Returns:
        a double, the value at time t on the gaussian with given parameters
    '''
    return amp*np.exp(-(t-Tc)**2/(2*sigma**2))

cpdef np.ndarray Map(fun, vec, args, dtype=np.float):
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

cdef class EhrenfestSU2:
    cdef dict __dict__
    def __init__(self, omega1, omega2, omega3, omega1_args, omega2_args, omega3_args):
        self.omega1 = omega1
        self.omega1_args = omega1_args
        self.omega2 = omega2
        self.omega2_args = omega2_args
        self.omega3 = omega3
        self.omega3_args = omega3_args

    @cython.boundscheck(False)
    def mixingAngles(self, np.ndarray[dtype=double, ndim=1, negative_indices=False] t):
        cdef int n = len(t)
        cdef np.ndarray alpha = np.zeros(n)
        cdef np.ndarray beta = np.zeros(n)
        cdef int i = 0
        cdef float omega_1, omega_2, omega_3, omega
        for i in range(n):
            omega_1 = self.omega1(t[i], *self.omega1_args)
            omega_2 = self.omega2(t[i], *self.omega2_args)
            omega_3 = self.omega3(t[i], *self.omega3_args)
            omega = sqrt(abs(omega_1) ** 2 + abs(omega_2) ** 2 + abs(omega_3) ** 2)
            alpha[i] = acos(abs(omega_3)/abs(omega))
            beta[i] = np.arctan2(abs(omega_1), abs(omega_2))
        return (alpha, beta)

    @cython.boundscheck(False)
    def mixingAnglesDerivative(self, double tt):
        t_step = self.t[1]-self.t[0]
        mixing_angles_1 = self.mixingAngles(np.array([tt - t_step / 2]))
        mixing_angles_2 = self.mixingAngles(np.array([tt + t_step / 2]))
        cdef double alpha1 = mixing_angles_1[0][0]
        cdef double alpha2 = mixing_angles_2[0][0]
        cdef double beta1 = mixing_angles_1[1][0]
        cdef double beta2 = mixing_angles_2[1][0]
        return ((alpha2-alpha1)/t_step, (beta2-beta1)/t_step)


    @cython.boundscheck(False)
    def _eom(self, np.ndarray[np.float_t, ndim=1, negative_indices=False] y, double t, double d1 = 0, double d2 = 0, double d3 = 0):
        mixing_angles = self.mixingAngles(np.array([t]))
        cdef double alpha = mixing_angles[0][0]
        cdef double beta = mixing_angles[1][0]
        alphaPrime, betaPrime = self.mixingAnglesDerivative(t)
        cdef double A11x = (1 + cos(beta) ** 2)
        cdef double A12x = 0.5* cos(alpha) * sin(2*beta)
        cdef double A22x = (cos(alpha)**2) * (1+sin(beta)**2)
        cdef double A11y = sin(beta) ** 2
        cdef double A12y = -0.5 * cos(alpha) * sin(2*beta)
        cdef double A22y = (cos(alpha) ** 2) * cos(beta)**2
        cdef double a_x = (A11x + A22x) / 2.0
        cdef double b_x = (A11x - A22x) / 2.0
        cdef double c_x = A12x
        cdef double a_y = (A11y + A22y) / 2.0
        cdef double b_y = (A11y - A22y) / 2.0
        cdef double c_y = A12y
        cdef np.ndarray Ax = (a_x * Id + b_x * sigz + c_x * sigx)
        cdef np.ndarray Ay = (a_y * Id + b_y * sigz + c_y * sigx)
        cdef np.ndarray A2 = np.matmul(Ax, Ax) + np.matmul(Ay, Ay)
        cdef double S11 = (cos(beta) ** 2 + 1) - A2[0, 0] / 2.0
        cdef double S22 = cos(alpha) ** 2 * (1 + sin(beta) ** 2) - A2[1, 1] / 2.0
        cdef double S12 = 0.5 * cos(alpha) * sin(2 * beta) - A2[0, 1] / 2.0
        cdef double D11 = (d1 + d2) / 2.0
        cdef double D12 = (sqrt(3) / 6) * (d1 - d2)
        cdef double D22 = (d1 + d2 + 4 * d3) / 6.0
        # a_s = (S11 + S22) / 2 + (D11 + D22) / 2 + (A2[0, 0] + A2[1, 1]) / 4
        cdef double b_s = (S11 - S22) / 2.0 + (D11 - D22) / 2.0 + (A2[0, 0] - A2[1, 1]) / 4.0
        cdef double c_s = S12 + D12 + A2[0, 1] / 2.0
        return np.array([y[5] - (a_x + b_x * y[4] + c_x * y[2]),
                         y[6] - (a_y + b_y * y[4] + c_y * y[2]),
                         2 * (b_x * y[5] + b_y * y[6] - b_s) * y[3],
                         -2 * (b_x * y[5] + b_y * y[6] - b_s) * y[2] + 2 * (c_x * y[5] + c_y * y[6] - c_s) * y[4],
                         -2 * (c_x * y[5] + c_y * y[6] - c_s) * y[3],
                         0,
                         0])

    @cython.boundscheck(False)
    def evolve(self, np.ndarray[double, ndim=1, negative_indices=False] t, np.ndarray[double, ndim=1, negative_indices=False] y0, double T, int N = 1000, double p_x = 0, double p_y = 0, double d1 = -1, double d2 = 1, double d3 = 3):
        cdef np.ndarray[double, ndim=1, negative_indices=False] px = sqrt(T) * np.random.randn(N) + p_x
        cdef np.ndarray[double, ndim=1, negative_indices=False] py = sqrt(T) * np.random.randn(N) + p_y
        cdef np.ndarray[double, ndim=1, negative_indices=False] x_0 = sqrt(T / (30/9600.0) ** 2) * np.random.randn(N)
        cdef np.ndarray[double, ndim=1, negative_indices=False] y_0 = sqrt(T / (60/9600.0) ** 2) * np.random.randn(N)
        cdef np.ndarray[double, ndim=3, negative_indices=False] raw = np.zeros((N, len(y0), len(t)), dtype=np.float)
        cdef np.ndarray[double, ndim=2, negative_indices=False] arg = np.zeros((N, len(y0)), dtype=np.float)
        cdef np.ndarray[double, ndim=2, negative_indices=False] y
        self.t = t
        self.d1 = d1/2
        self.d2 = d2/2
        self.d3 = d3/2
        for i in range(N):
            y0[0], y0[1] =  x_0[i], y_0[i]
            y0[5], y0[6] = px[i], py[i]
            y = odeint(self._eom, y0, self.t, args=(self.d1, self.d2, self.d3))
            raw[i] = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], y[:, 6]#'''
        self.raw = raw
        cdef np.ndarray[double, ndim=2, negative_indices=False] result = np.mean(self.raw, axis=0)
        self.result = result
        return self.result

    @cython.boundscheck(False)
    def bareStatePop(self, np.ndarray[np.float_t, ndim=1, negative_indices=False] t):
        alpha, beta = self.mixingAngles(t)
        x, y, sx, sy, sz, px, py = self.result
        P1 = (1 + sz) / 2 * cos(beta) ** 2 + (1 - sz) / 2 * cos(alpha) ** 2 * sin(beta) ** 2 + sx * cos(alpha) * cos(beta) * sin(beta)
        P2 = (1 + sz) / 2 * sin(beta) ** 2 + (1 - sz) / 2 * cos(alpha) ** 2 * cos(beta) ** 2 - sx * cos(alpha) * cos(beta) * sin(beta)
        P3 = (1 - sz) / 2 * sin(alpha) ** 2
        return np.array([P1, P2, P3])

    @cython.boundscheck(False)
    def bareStatePop2(self, np.ndarray[np.float_t, ndim=1, negative_indices=False] t, result):
        alpha, beta = self.mixingAngles(t)
        cdef int n = len(t)
        cdef np.ndarray P1 = np.zeros(n)
        cdef np.ndarray P2 = np.zeros(n)
        cdef np.ndarray P3 = np.zeros(n)
        for i in range(n):
            sx, sy, sz = result[2:5, i]
            P1[i] = (1 + sz) / 2 * cos(beta[i]) ** 2 + (1 - sz) / 2 * cos(alpha[i]) ** 2 * sin(beta[i]) ** 2 + sx * cos(alpha[i]) * cos(beta[i]) * sin(beta[i])
            P2[i] = (1 + sz) / 2 * sin(beta[i]) ** 2 + (1 - sz) / 2 * cos(alpha[i]) ** 2 * cos(beta[i]) ** 2 - sx * cos(alpha[i]) * cos(beta[i]) * sin(beta[i])
            P3[i] = (1 - sz) / 2 * sin(alpha[i]) ** 2
        return np.array([P1, P2, P3])


cdef class EhrenfestSU3:
    cdef dict __dict__
    def __init__(self, omegas, omega_args):
        self.omegas = omegas
        self.omega_args  = omega_args

    def mixingAngles(self, np.ndarray t):
        pass

    def _eom(self):
        pass

    def evolve(self):
        pass

    def bareStatePop(self):
        pass

