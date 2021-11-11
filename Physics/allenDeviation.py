import numpy as np
from math import *
from ..picoMatTools import picoMatRead

class TSeries(object):
    def __init__(self, dataPath, channel, f0, Amplitude=None, offset=None):
        data = picoMatRead(dataPath, channels=[channel])
        self.time = data[0][0] # time axis
        self.f0 = f0 # nominal frequency of the ref. osc.
        self.sampleTime = self.time[1]-self.time[0]
        if Amplitude==None:
            if offset==None:
                beat = data[1][0] - np.mean(data[1][0])
                max_sig = np.max(abs(beat))
                phase = np.arccos(beat/max_sig)
                self.angle = np.unwrap(phase) # phase fluctuation
        else:
            beat = data[1][0] - offset
            phase = np.arccos(beat/Amplitude)
            self.angle = np.unwrap(phase)
    def fracFreq(self):
        deltaFreq = np.gradient(self.angle, self.time)/(2*pi) # freq. fluctuation
        return deltaFreq/self.f0
    def timeDiff(self):
        self.x = self.angle/(2*pi*self.f0) # time difference
        return self.x
    def allenVar(self, m):
        x = self.timeDiff()
        tau = self.time[m]-self.time[0] #sampling time in sec
        y = np.zeros(int(len(x)/m))
        for i in range(len(x)//m-1):
            y[i] = (x[m*i+m]-x[m*i])/tau
        s = 0
        for i in range(len(y)-1):
            s += (y[i+1]-y[i])**2
        return s/(2*(len(y)-1))
    def allenDev(self, m):
        return np.sqrt(self.allenVar(m))
    def oAllenVar(self, m):
        x = self.timeDiff()
        tau = self.time[m]-self.time[0] #sampling time in sec
        y = np.zeros(len(x)-m)
        for i in range(len(x)-m-1):
            y[i] = (x[i+m]-x[i])/tau
        s = 0
        for i in range(len(y)-1):
            s += (y[i+1]-y[i])**2
        return s/(2*(len(y)-1))
    def oAllenDev(self, m):
        return np.sqrt(self.oAllenVar(m))