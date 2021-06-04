import pyvisa as v
import numpy as np

class FuncGen(object):
    def __init__(self, connectionString):
        rm = v.ResourceManager()
        try:
            self.ins = rm.open_resource(connectionString)
            print('Device found!')
        except v.VisaError:
            print('Device not found')

    def arbBurst(self, waveform, frequency, channel=1):
        wavestring = ''
        for i, number in enumerate(waveform):
            wavestring += str(np.round(number, 4))+','
        waveString = wavestring[:-1]
        if channel == 1:
            source = 'Source1'
        elif channel == 2:
            source = 'Source2'
        else:
            raise ValueError('Source not defined! Check the channel number')
        func = self.ins
        func.write(':'+source+':TRACE:DATA VOLATILE,' + waveString)
        func.write(':'+source+':Frequency '+ str(frequency))
        func.write(':'+source+':Function:ARB')
        func.write(':'+source+':Burst:Ncycles 1')
        func.write(':'+source+':Burst:Mode Trig')
        func.write(':'+source+':Burst:Trigger:Source External')
        func.write(':'+source+':Burst ON')
        func.write(':Output'+str(channel)+' ON')

    def switch(self, channel, status):
        key = ['OFF', 'ON']
        self.ins.write(':Output' + str(int(channel)) + ' ' + key[status])

if __name__=="__main__":
    FuncGen('TCPIP0::192.168.200.122::inst0::INSTR')
