from multiprocessing import Process
import numpy as np
from ehrenfest import EhrenfestSU2, omegaConstant, omegaRamp, omegaGaussian
import matplotlib.pyplot as  plt
import time
import sys
import os



def run_ehrenfest(t_r, px, py):
    t_unit = 16.576
    y0 = np.array([0, 0, 0, 0, -1, 0, 0], dtype=float)
    t_vec = np.linspace(0, 100/t_unit, 200, dtype=float)
    t_ramp = float(t_r)/t_unit
    e = EhrenfestSU2(omegaConstant, omegaConstant, omegaRamp, (), (), (t_ramp,))
    x = e.evolve(t_vec, y0, T=0.1, N=1000, p_x=float(px), p_y=float(py), d1=1, d2=-1,d3=3)
    np.savetxt('result'+str(os.getpid())+'.txt', x)

def run_ehrenfest_sfet(sigma, factor):
    t_unit = 16.576
    y0 = np.array([0, 0, 0, 0, -1, 0, 0], dtype=float)
    t_vec = np.linspace(0, 100/t_unit, 200, dtype=float)
    tc1 = 3*sigma
    tc2 = tc1+factor*sigma
    tc3 = tc2+factor*sigma
    e = EhrenfestSU2(omegaGaussian, omegaGaussian, omegaGaussian, (1, tc1, sigma), (1, tc2, sigma), (1, tc3, sigma))
    x = e.evolve(t_vec, y0, T=0.1, N=250, d1=0,d2=0,d3=0)
    np.savetxt('result'+str(os.getpid())+'.txt', x)


if __name__=='__main__':
    t_r, px, py = map(float, sys.argv[1:])
    #sigma, factor = map(float, sys.argv[1:])
    t1= time.time()
    processes = []
    for i in range(1):
        p = Process(target=run_ehrenfest, args=(t_r, px, py))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    result = []
    for file in os.listdir():
        if file.split('.')[-1] == 'txt':
            result.append(np.loadtxt(file))
            os.remove(file)
    result = np.array(result)
    result = np.mean(result, axis=0)
    np.savetxt('Result.txt', result)
    t2 = time.time()
    print('Total time: '+str(t2-t1)+' s')
    r = np.loadtxt('Result.txt')
    t_unit = 16.576
    y0 = np.array([0, 0, 0, 0, -1, 0, 0], dtype=float)
    t_vec = np.linspace(0, 100/t_unit, 200, dtype=float)
    '''
    tc1 = 3 * sigma
    tc2 = tc1 + factor * sigma
    tc3 = tc2 + factor * sigma'''
    #e = EhrenfestSU2(omegaGaussian, omegaGaussian, omegaGaussian, (1, tc1, sigma), (1, tc2, sigma), (1, tc3, sigma))
    e = EhrenfestSU2(omegaConstant, omegaConstant, omegaRamp, (), (), (t_r,))
    P1, P2, P3 = e.bareStatePop2(t_vec, r)
    plt.figure()
    #plt.plot(t_vec, r[4])
    plt.plot(t_vec*t_unit, P1)
    plt.plot(t_vec*t_unit, P2)
    plt.plot(t_vec*t_unit, P3)
    plt.show()