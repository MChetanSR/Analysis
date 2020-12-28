from multiprocessing import Pool, Process
import numpy as np
from ehrenfest_cython import Ehrenfest, omegaConstant, omegaRamp
import time
import sys

def run_ehren(t_r, px, py):
    t_unit = 16.576
    y0 = np.array([0, 0, 0, 0, -1, 0, 0], dtype=float)
    t_ramp = float(t_r)/t_unit
    t_vec = np.linspace(0, 100/t_unit, 200, dtype=float)
    e = Ehrenfest(omegaConstant, omegaConstant, omegaRamp, (), (), (t_ramp,))
    x = e.evolve(t_vec, y0, T=0.1, N=250, p_x=float(px), p_y=float(py))

if __name__=='__main__':
    t_r, px, py = sys.argv[1:]
    processes = []
    t1= time.time()
    for i in range(4):
        process = Process(target=run_ehren, args=(t_r, px, py))
        processes.append(processes)
        process.start()
        #process.join()
    process.join()
    t2 = time.time()
    print('Total time: '+str(t2-t1))