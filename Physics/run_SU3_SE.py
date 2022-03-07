from multiprocessing import Process
import numpy as np
from ehrenfest import  SU3_GaugeField, mixingAngles
from scipy.constants import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as  plt
import time
import sys
import os

def SE(t, y, px, py, t_ramp, scalar=0):
    omega_r = 2 * pi * 4.78 * milli
    if t<t_ramp:
        x = (1/t_ramp)*t
        Dmix = np.array([[0, 0, 0], [0, 0, -1j*pi/4/t_ramp], [0, 1j*pi/4/t_ramp, 0]])
    else:
        x = 1
        Dmix = np.zeros((3, 3))
    theta_l, phi_l = mixingAngles(1, 1, x)
    theta_r, phi_r = mixingAngles(x, 1, 1)
    alpha = np.sqrt(1+np.tan(theta_l)**-2+np.tan(theta_r)**-2)
    Ax, Ay = SU3_GaugeField(theta_l, phi_l, theta_r, phi_r)
    W = 2*np.array([[3*np.sin(phi_l)**2+5, 1.5*np.sin(2*phi_l)/alpha/np.tan(theta_l), 0],
                    [1.5*np.sin(2*phi_l)/alpha/np.tan(theta_l),
                     ((16-6*np.sin(phi_l)**2)/np.tan(theta_l)**2 + 2*np.sin(phi_r)**2/np.tan(theta_r)**2+2)/(2*alpha**2),
                     -0.5*np.sin(2*phi_r)/alpha/np.tan(theta_r)],
                    [0, -0.5*np.sin(2*phi_r)/alpha/np.tan(theta_r), np.cos(phi_r)**2]])
    return -1j*((px**2+py**2)*omega_r*np.eye(3) -
            (px*Ax + py*Ay)*2*omega_r -
            (np.tan(theta_r)**-1/alpha)*Dmix +
            W*scalar*omega_r)@y

def thermalAverage(t, px, py, T, N, t_ramp, scalar):
    omega_r = 2 * pi * 4.78 * milli
    T = T * 130.92 / (2 * omega_r * 1e6)
    px_ran = px + np.random.randn(N) * np.sqrt(T)
    py_ran = py + np.random.randn(N) * np.sqrt(T)
    result = np.zeros((3, len(t)), dtype=float)
    for i in range(N):
        pxr, pyr = px_ran[i], py_ran[i]
        sol = solve_ivp(SE, (t[0], t[-1]), y0=[0, 0, 1 + 0j], args=(pxr, pyr, t_ramp, scalar), method='RK45', t_eval=t)
        result += abs(sol.y)**2/N
    np.savetxt('result'+str(os.getpid())+'.txt', result.flatten())

if __name__=='__main__':
    plt.style.use(os.path.abspath('../Styles/Helvetica_bright.mplstyle'))
    if 'Result.txt' in os.listdir():
        os.remove('Result.txt')
    px, py = map(float, sys.argv[1:])
    t = np.linspace(0, 150, 300, endpoint=False)
    omega_r = 2 * pi * 4.78 * milli
    t_ramp = 16
    scalar = 0
    T = 50
    N = 250
    t1= time.time()
    processes = []
    for i in range(4):
        p = Process(target=thermalAverage, args=(t, px, py, T, N, t_ramp, scalar))
        p.start()
        print('Process: '+str(p.pid)+' started')
        processes.append(p)
    for p in processes:
        p.join()
    result = []
    for file in os.listdir():
        if file.split('.')[-1] =='txt':
            array = np.genfromtxt(file)
            result.append(array)
            os.remove(file)
    result = np.array(result)
    result = np.mean(result, axis=0)
    np.savetxt('../../Result.txt', result)
    t2 = time.time()
    print('Total time: '+str(t2-t1)+' s')
    r = np.loadtxt('../../Result.txt')
    #e = EhrenfestSU2(omegaConstant, omegaConstant, omegaRamp, (), (), (t_r/t_unit,))
    #P1, P2, P3 = e.bareStatePop2(t_vec, r)
    plt.figure()
    plt.plot(t, r.reshape((3, len(t)))[0])
    plt.plot(t, r.reshape((3, len(t)))[1])
    plt.plot(t, r.reshape((3, len(t)))[2])
    #plt.show()

