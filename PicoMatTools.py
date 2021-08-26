import os
from scipy.io import loadmat
from numpy.fft import fft, fftfreq
import numpy as np

def picoMatRead(filePath, channels=['A']):
    file = loadmat(filePath)
    tStep = file['Tinterval']
    N = file['Length'][0, 0]
    t = np.arange(0,N)*tStep
    result = np.zeros((len(channels), N))
    for i, ch in enumerate(channels):
        result[i] = file[ch][:,0]
    return t, result


def PSD(path, avg=30, channels=['A']):
    files = os.listdir(path)
    first = loadmat(os.path.join(path, files[0]))
    tStep = first['Tinterval']
    N = first['Length'][0,0]
    frequencies = fftfreq(N, tStep)[0,:int(N/2)]
    result = np.zeros((len(channels), int(N/2)))
    for f in files[:avg]:
        data = loadmat(os.path.join(path, f))
        for i, ch in enumerate(channels):
            result[i] += 2*abs(fft(data[ch][:,0])[:N//2]/N)**2
    result /= avg
    result[0] = result[0]/2
    return frequencies, result


def RIN(path, channel='A'):
    files = os.listdir(path)
    first = loadmat(os.path.join(path, files[0]))
    tStep = first['Tinterval']
    N = first['Length'][0,0]
    frequencies = fftfreq(N, tStep)[0,:int(N/2)]
    inLoop = []
    for f in files[:-2]:
        data = loadmat(os.path.join(path, f))
        inLoop.append(2*abs(fft(data[channel][:,0])[:N//2]/N)**2/np.mean(data[channel][:, 0])**2)
        inLoopRIN = np.mean(np.array(inLoop), axis=0)
        inLoopRIN[0], inLoopRIN[-1] = inLoopRIN[0]/2, inLoopRIN[-1]/2
    return frequencies, 10*np.log10(inLoopRIN)

