from multiprocessing import Process
import numpy as np
from .ehrenfest_cython import EhrenfestSU2, omegaConstant, omegaRamp, omegaGaussian, Map
import matplotlib.pyplot as  plt
import time
import sys
import os
#plt.style.use('./Styles/DarkBackground.mplstyle')

def run_ehrenfest(t_r, px, py):
    t_unit = 16.576
    y0 = np.array([0, 0, 0, 0, -1, 0, 0], dtype=float)
    t_vec = np.linspace(0, 100/t_unit, 400, dtype=float)
    t_ramp = float(t_r)/t_unit
    e = EhrenfestSU2(omegaConstant, omegaConstant, omegaRamp, (), (), (t_ramp,))
    x = e.evolve(t_vec, y0, T=0.13, N=250, p_x=float(px), p_y=float(py), d1=-1, d2=1,d3=3)
    np.savetxt('result'+str(os.getpid())+'.txt', x)

if __name__=='__main__':
    plt.style.use(os.path.abspath('../Styles/Helvetica_bright.mplstyle'))
    if 'Result.txt' in os.listdir():
        os.remove('Result.txt')
    t_r, px, py = map(float, sys.argv[1:])
    t1= time.time()
    processes = []
    for i in range(4):
        p = Process(target=run_ehrenfest, args=(t_r, px, py))
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
    result = np.array(result, dtype=float)
    result = np.mean(result, axis=0)
    np.savetxt('Result.txt', result)
    t2 = time.time()
    print('Total time: '+str(t2-t1)+' s')
    r = np.loadtxt('Result.txt')
    t_unit = 16.576
    y0 = np.array([0, 0, 0, 0, -1, 0, 0], dtype=float)
    t_vec = np.linspace(0, 100/t_unit, 400, dtype=float)
    e = EhrenfestSU2(omegaConstant, omegaConstant, omegaRamp, (), (), (t_r/t_unit,))
    P1, P2, P3 = e.bareStatePop2(t_vec, r)
    plt.figure()
    plt.plot(t_vec*t_unit, P1)
    plt.plot(t_vec*t_unit, P2)
    plt.plot(t_vec*t_unit, P3)
    plt.show()
    plt.figure()
    plt.scatter(result[:, 0], result[:, 1])
    plt.show()
