# Noise curve taken from LDC used for Sangria and Spritz, "MRDv1"
from abc import ABC, abstractmethod
#import cupy as cp
CLIGHT = 299792458. 
arm_length = 2.5e9

cuda = 1

# Making code agnostic to CPU/GPU
def std_get_wrapper(arg):
    return arg

def cuda_get_wrapper(arg):
    return arg.get()

if cuda:
   import cupy as xp
   gpu = True
   get_wrapper = cuda_get_wrapper
   dev = "cuda:0"
else:
   import numpy as xp
   gpu = False
   get_wrapper = std_get_wrapper
   dev = "xpu"

# Create samples of the noise with the defined variance
def sample_noise(variance, df):

   n_real = xp.random.normal(loc=0.0, scale=xp.sqrt(variance)/(xp.sqrt(4.0*df)))
   n_imag = xp.random.normal(loc=0.0, scale=xp.sqrt(variance)/(xp.sqrt(4.0*df)))

   return n_real+1j*n_imag
  

class AnalyticNoise(ABC):
    """Analytic approximation of the two components of LISA noise:
    acceleration noise and optical metrology system (OMS) noise
    """
    
    def __init__(self, frq, noise_type):
        """Set two components noise contributions wrt given model.
        """
        super().__init__()
        
        if noise_type == 'MRDv1':
            self.DSoms_d = (10.e-12)**2  # m^2/Hz
            self.DSa_a = (2.4e-15)**2 # m^2/sec^4/Hz
        elif noise_type == 'sangria':
            self.DSoms_d = (7.9e-12)**2 # m^2/Hz
            self.DSa_a = (2.4e-15)**2 # m^2/sec^4/Hz
        else:
            raise NotImplementedError('Noise type '+ noise_type + ' not implemented')
        self.freq = frq            

        # Acceleration noise
        Sa_a = self.DSa_a * (1.0 +(0.4e-3/frq)**2) * (1.0+(frq/8e-3)**4) # in acceleration
        self.Sa_d = Sa_a*(2.*xp.pi*frq)**(-4.) # in displacement
        Sa_nu = self.Sa_d*(2.0*xp.pi*frq/CLIGHT)**2 # in rel freq unit
        self.Spm =  Sa_nu

        # Optical Metrology System
        self.Soms_d = self.DSoms_d * (1. + (2.e-3/frq)**4) # in displacement
        Soms_nu = self.Soms_d*(2.0*xp.pi*frq/CLIGHT)**2 # in rel freq unit
        self.Sop =  Soms_nu

    def psd(self, option="A", tdi2=False):#, includewd=0):
        """ Return noise PSD at given freq. or freq. range.

        Option can be X, A, E, T.

        """
        lisaLT = arm_length/CLIGHT
        x = 2.0 * xp.pi * lisaLT * self.freq #xp.array(self.freq)
       
        if option=="X":
            S = 16.0 * xp.sin(x)**2 * (2.0 * (1.0 + xp.cos(x)**2) * self.Spm + self.Sop)
        elif option in ["A", "E"]:
            S = 8.0 * xp.sin(x)**2  * (2.0 * self.Spm * (3.0 + 2.0*xp.cos(x) + xp.cos(2.0*x)) + self.Sop * (2.0 + xp.cos(x)))
        elif option=="T":
            S = 16.0 * self.Sop * (1.0 - xp.cos(x)) * xp.sin(x)**2 + 128.0 * self.Spm * xp.sin(x)**2 * xp.sin(0.5*x)**4
        else:
            print("PSD option should be in [X, A, E, T] (%s)"%option)
            return None
        if tdi2:
            factor_tdi2 = 4 * xp.sin(2 * x)**2
            S *= factor_tdi2
        return S