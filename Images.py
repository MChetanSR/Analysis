import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from warnings import warn
from .AndorSifReader import AndorSifFile
from .fits import gaussian2DFit
from scipy.constants import *


class ShadowImage(object):
    """
    The main class to extract relavent information from the shadow image
    sequences that are obtained in the experiment. The multiple image sequence
    consists of the shadow or the absorption image, image of the incident probe
    and the image of the background for as many runs of the experiment.
    
    Parameters:
        filePath: str, Path to the multiple-image file

    Attributes:
        filePath: str, Path to the multiple-image file
        im: PIL.TiffImagePlugin.TiffImageFile or AndorSifFile._SifFrames
        n: int, total number of data = number of images/3
        frames: ndarray, all the frames in the image file
        tags: dict, the named tag dictionary of the TiffImage or properties of sif file
        transmission: ndarray, All the transmission images after
                       subtracting the background
        incidence: ndarray, All the images of incident probe after
                    subtracting the background
        OD: ndarray, optical depth all the runs of the experiment calculated from
             the transmission and incidence
        ODaveraged: ndarray, OD averaged with averaging like a superloop
        averagedOD: ndarray, OD of the averaged signal with averaging on superloop
        averagedOD2: ndarray, OD of the averaged signal with averaging on loop
    """
    def __init__(self, filePath):
        self.filePath = filePath
        self.ext = os.path.splitext(filePath)[1]
        if self.ext == '.tif':
            self.im = Image.open(filePath)
            self.tags = self.im.tag.named()
        elif self.ext == '.sif':
            self.im = AndorSifFile(filePath).signal
            self.tags = self.im.props
        else:
            raise IOError('Invalid file!')
        if self.im.n_frames%3!=0:
            warn('Not a valid shadow image. \
                  No. of images in the file is not a multiple of 3.')
        self.n = self.im.n_frames//3
        self.frames = self.images()

    def images(self):
        """
        Returns the images present in the ShadowImage object as an ndarray.
        """
        if self.ext == '.tif':
            frames = np.zeros((self.im.n_frames, self.im.height, self.im.width))
            for i in range(self.im.n_frames):
                self.im.seek(i)
                frames[i] = np.array(self.im)
        elif self.ext == '.sif':
            frames = self.im.data
        return frames

    def opticalDepth(self, xSpan, ySpan):
        """
        Returns the optical depth calculated from the ShadowImage as an ndarray.
        xSpan: a list containing the range of pixels of the image in x direction.
        ySpan: a list containing the range of pixels of the image in y direction.
        """
        self.transmission = np.zeros((self.n, self.im.height, self.im.width))
        self.incidence = np.zeros((self.n, self.im.height, self.im.width))
        for i in range(self.n):
            self.transmission[i] = self.frames[3*i]-self.frames[3*i+2]
            self.incidence[i] = self.frames[3*i+1]-self.frames[3*i+2]
        self.incidence[self.incidence == 0] = 1e-5
        self.incidence[self.incidence != self.incidence] = 1e-5
        self.transmission[self.transmission == 0] = 1e-5
        self.transmission[self.transmission != self.transmission] = 1e-5
        T = self.transmission[:,ySpan[0]:ySpan[1], xSpan[0]:xSpan[1]]\
                               /self.incidence[:,ySpan[0]:ySpan[1], xSpan[0]:xSpan[1]]
        T[T <= 0] = 1e-20
        T[T != T] = 1e-20
        #T[T > 1+1e-3] = 1
        self.OD = -np.log(T)
        return self.OD

    def opticalDepthAveraged(self, nAveraging):
        """
        Calculates the optical depth from every triad of the images and
        averages it with nAveraging as the Superloop in the experiment.
        Returns an ndarray of length equal to Serie in the experiment.
        """
        self.nAveraging = nAveraging
        self.nSamples = int(self.n/nAveraging)
        result = np.zeros((self.nSamples, self.im.height, self.im.width))
        ShadowImage.opticalDepth(self, [0, self.im.width], [0, self.im.height])
        for j in range(self.nSamples):
            for i in range(self.nAveraging):
                result[j] += self.OD[self.nSamples*i+j]
            result[j] /= self.nAveraging
        self.ODaveraged = result
        return self.ODaveraged

    def averagedSignalOD(self, nAveraging, truncate=[]):
        """
        Calculates the average signal with nAveraging being the Superloop
        in the experiment and finds the optical depth after the averaging.
        Returns an ndarray of length equal to Serie in the experiment.
        """
        self.nAveraging = nAveraging
        self.nSamples = int(self.n/nAveraging)
        self.averagedTransmission = np.zeros((self.nSamples, self.im.height, self.im.width))
        self.averagedIncidence = np.zeros((self.nSamples, self.im.height, self.im.width))
        ShadowImage.opticalDepth(self, [0, self.im.height], [0, self.im.height])
        for j in range(self.nSamples):
            for i in range(self.nAveraging):
                if not(i in truncate):
                    self.averagedTransmission[j] += self.transmission[self.nSamples*i+j]
                    self.averagedIncidence[j] += self.incidence[self.nSamples*i+j]
        self.averagedTransmission /= (nAveraging-len(truncate))
        self.averagedIncidence /= (nAveraging-len(truncate))
        self.averagedIncidence[self.averagedIncidence == 0] = 1e-5        
        T = self.averagedTransmission/self.averagedIncidence
        T[T != T] = 1e-20
        T[T <= 0] = 1e-20
        self.averagedOD = -np.log(T)
        return self.averagedOD

    def averagedSignalOD2(self, nAveraging):
        """
        Calculates the average signal with nAveraging being the loops
        in the experiment and finds the optical depth after the averaging.
        Returns an ndarray of length equal to Serie in the experiment.
        """
        self.nAveraging = nAveraging
        self.nSamples = int(self.n/nAveraging)
        self.averagedTransmission2 = np.zeros((self.nSamples, self.im.height, self.im.width))
        self.averagedIncidence2 = np.zeros((self.nSamples, self.im.height, self.im.width))
        ShadowImage.opticalDepth(self, [0, self.im.height], [0, self.im.height])
        for j in range(self.nSamples):
            for i in range(self.nAveraging):
                self.averagedTransmission2[j] += self.transmission[self.nSamples*j+i]
                self.averagedIncidence2[j] += self.incidence[self.nSamples*j+i]
        
        self.averagedTransmission /= (nAveraging-len(truncate))
        self.averagedIncidence /= (nAveraging-len(truncate))
        self.averagedIncidence2[self.averagedIncidence == 0] = 1e-5
        T = self.averagedTransmission2/self.averagedIncidence2
        T[T != T] = 1e-20
        T[T <= 0] = 1e-20
        self.averagedOD2 = -np.log(T)
        return self.averagedOD2
       
    def plotAveragedSignalOD(self, nAveraging, ROI, truncate=[]):
        """
        Calculates and plots the average signal with nAveraging being the Superloop
        in the experiment and finds the optical depth after the averaging.
        Returns None.
        """
        OD = self.averagedSignalOD(nAveraging, truncate)
        f, ax = plt.subplots(nrows=len(OD), ncols=1, figsize=(4,len(OD)*2))
        for i in range(len(OD)):
            a = ax[i].imshow(OD[i, ROI[0]:ROI[1], ROI[2]:ROI[3]], cmap=plt.cm.hot)
            ax[i].grid(False)
            f.colorbar(a, ax=ax[i])
        plt.tight_layout()
        return None
        
        
    '''
    def comment(self, comment):
        """
        Adds comment to the tif image under the ImageDescription tag
        and replaces it silently with the new image, if the user has the
        permissions.
        """
        path, ext = os.path.splitext(self.filePath)
        newPath = path+'cmtd'+ext
        self.im.save(newPath, save_all=True, description=str(comment))
        os.replace(newPath, self.filePath)

    def description(self):
        """
        Returns the ImageDescription tag of the tiff image file.
        """
        try:
            r = self.tags['ImageDescription']
        except KeyError:
            r = ''
        return r
    '''
    def redProbeIntensity(self, params, plot=False):
        """
        Returns intensity(micro Watt/cm^2), power(W), waist_x, waist_y
        """
        try:
            probeImage = self.averagedIncidence[0]
        except AttributeError:
            print("Call averagedSignalOD to calculate averagedIncidece before calling probeIntensity.")
            return
        y = self.im.height
        x = self.im.width
        imConstant = params['binning']*params['pixelSize']/params['magnification']
        pOpt, pCov = gaussian2DFit(probeImage, p0=[4000, x/2, y/2, x/2, y/2, np.pi/4, 500], plot=plot)
        totalCount = np.sum(np.sum(probeImage))
        wx = abs(2*pOpt[3]*imConstant*100) # in cm
        wy = abs(2*pOpt[4]*imConstant*100) # in cm
        area = np.pi*wx*wy
        if self.ext=='.tif': # for PCO panda 4.2 bi camera red imaging
            photons = totalCount*0.8/(0.85)
            energy = photons*h*c/(689*nano)
            power = energy/(0.95*60*micro) # 0.95 to account for filter and losses on optics
            intensity = 2*(power/1e-6)/(area) # in micro Watt/cm^2
            return intensity, power, wx, wy
        elif self.ext == '.sif': # for andor
            raise NotImplementedError
    
    def blueProbeIntensity(self, params, plot=False):
        """
        Returns intensity(micro Watt/cm^2), power(W), waist_x, waist_y
        """
        try:
            probeImage = self.averagedIncidence[0]
        except AttributeError:
            print("Call averagedSignalOD to calculate averagedIncidece before calling probeIntensity.")
            return
        y = self.im.height
        x = self.im.width
        imConstant = params['binning']*params['pixelSize']/params['magnification']
        pOpt, pCov = gaussian2DFit(probeImage, p0=[4000, x/2, y/2, x/2, y/2, np.pi/4, 500], plot=plot)
        totalCount = np.sum(np.sum(probeImage))
        wx = abs(2*pOpt[3]*imConstant*100) # in cm
        wy = abs(2*pOpt[4]*imConstant*100) # in cm
        area = np.pi*wx*wy
        if self.ext=='.tif': # for PCO panda 4.2 bi camera blue imaging
            photons = totalCount*0.8/(0.85)
            energy = photons*h*c/(461*nano)
            power = energy/(0.95*20*micro) # 0.95 to account for filter and losses on optics
            intensity = 2*(power/1e-6)/(area) # in micro Watt/cm^2
            return intensity, power, wx, wy
        elif self.ext == '.sif': # for andor
            raise NotImplementedError

    def __str__(self):
        return str(self.tags)

class FluorescenceImage(object):
    """
    The main class to extract relavent information from the fluorescence image
    sequences that are obtained in the experiment. The multiple image sequence
    consists of fluorecence signal from the atomic cloud for as many runs of the
    experiment.
    
    Parameters:
        filePath: str, Path to the multiple-image file
    """
    def __init__(self, filePath):
        self.filePath = filePath
        self.ext = os.path.splitext(filePath)[1]
        if self.ext == '.tif':
            self.im = Image.open(filePath)
            self.tags = self.im.tag.named()
        elif self.ext == '.sif':
            self.im = AndorSifFile(filePath).signal
            self.tags = self.im.props
        else:
            raise IOError('Invalid file!')
        if self.im.n_frames%2!=0:
            warn('Not a valid fluorescence image. \
                  No. of images in the file is not a multiple of 2.')
        self.n = self.im.n_frames//2
        self.frames = self.images()

    def images(self):
        """
        Returns the images present in the multiple image file as an ndarray.
        """
        if self.ext == '.tif':
            frames = np.zeros((self.im.n_frames, self.im.height, self.im.width))
            for i in range(self.im.n_frames):
                self.im.seek(i)
                frames[i] = np.array(self.im)
        elif self.ext == '.sif':
            frames = self.im.data
        return frames
    
    def fluorescence(self, xSpan, ySpan):
        """
        Returns the fluorence calculated from the FluorencenceImage as an ndarray.
        xSpan: a list containing the range of pixels of the image in x direction.
        ySpan: a list containing the range of pixels of the image in y direction.
        """
        self.fluorescence = np.zeros((self.n, self.im.height, self.im.width))
        for i in range(self.n):
            self.fluorescence[i] = self.frames[2*i]-self.frames[2*i+1]

        self.fluorescence[self.fluorescence <= 0] = 1e-5
        self.fluorescence[self.fluorescence != self.fluorescence] = 1e-5
        return self.fluorescence
    
    def averagedSignal(self, nAveraging, truncate=[]):
        """
        Calculates the average signal with nAveraging being the Superloop
        in the experiment and finds the fluorescence after the averaging.
        Returns an ndarray of length equal to Serie in the experiment.
        """
        self.nAveraging = nAveraging
        self.nSamples = int(self.n/nAveraging)
        self.averagedFluorescence = np.zeros((self.nSamples, self.im.height, self.im.width))
        self.averagedBackground = np.zeros((self.nSamples, self.im.height, self.im.width))
        FluorescenceImage.fluorescence(self, [0, self.im.width], [0, self.im.height])
        for j in range(self.nSamples):
            for i in range(self.nAveraging):
                if not(i in truncate):
                    self.averagedFluorescence[j] += self.fluorescence[self.nSamples*i+j]
                    self.averagedBackground[j] += self.frames[2*(self.nSamples*i+j)+1]
        self.averagedFluorescence[self.averagedFluorescence <= 0] = 1e-5  
        self.averagedBackground /= (nAveraging-len(truncate))
        return self.averagedFluorescence/(nAveraging-len(truncate))
    
    def plotAveragedSignal(self, nAveraging, ROI, truncate=[]):
        """
        Calculates and plots the average signal with nAveraging being the Superloop
        in the experiment.
        Returns None.
        """
        fl = self.averagedSignal(nAveraging, truncate)
        w = int(self.im.width/(self.im.width+self.im.height)*4)
        h = int(self.im.height/(self.im.width+self.im.height)*4)
        f, ax = plt.subplots(nrows=len(fl), ncols=1, figsize=(w, h))
        for i in range(len(fl)):
            a = ax[i].imshow(fl[i, ROI[0]:ROI[1], ROI[2]:ROI[3]], cmap=plt.cm.hot)
            ax[i].grid(False)
            f.colorbar(a, ax=ax[i])
        plt.tight_layout()
        return None
    
    
    def __str__(self):
        return str(self.tags)