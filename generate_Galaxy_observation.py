from gbgpu.gbgpu import GBGPU
from gbgpu.utils.constants import *
from noisemodel import *
import numpy as np
from matplotlib import pyplot as plt

cuda = 1

# Making code agnostic to CPU/GPU
def std_get_wrapper(arg):
    return arg

def cuda_get_wrapper(arg):
    return arg.get()

if cuda:
   gpu = True
   get_wrapper = cuda_get_wrapper
   dev = "cuda:0"
   import cupy as xp
else:
   gpu = False
   get_wrapper = std_get_wrapper
   dev = "cpu"
   import numpy as xp

dt = 10.0
Tobs = 1.0*YEAR
num = 128

# Load generated sample from the Galaxy
parameters_waveform =  xp.asarray(np.load('galaxy.npy'))
N = parameters_waveform.shape[1]

import time

start_time = time.time() 

gb = GBGPU(use_gpu=True)        
gb.run_wave(*parameters_waveform, N = num, dt = dt, T = Tobs, oversample=2)

wf_A = gb.A
wf_E = gb.E

'''
Generate noise from the PSD.
'''
# Frequencies from 1e-5 to 
df = 1./Tobs
#freq_min = 1e-4
#kmin = xp.floor(freq_min/dt).astype(int)
freq_max = 1./(2.*dt)
kmax = xp.floor(freq_max/df).astype(int)
freqs = xp.arange(kmax)*df

i_start = gb.start_inds.get().astype(int)
i_end   = (gb.start_inds.get() + gb.N).astype(int)

noise = AnalyticNoise(freqs, 'MRDv1')
noisevals_A, noisevals_E = noise.psd(option="A"), noise.psd(option="E")
     
noiseA = sample_noise(noisevals_A, df)
noiseE = sample_noise(noisevals_E, df)

'''
Add the noise to the signal, final realisation of the Galaxy
'''

xA_all = xp.zeros(freqs.shape, dtype = xp.complex128)
xE_all = xp.zeros(freqs.shape, dtype = xp.complex128)
for i in range(N):
    xA = xp.zeros(freqs.shape, dtype = xp.complex128)
    xE = xp.zeros(freqs.shape, dtype = xp.complex128)

    xA[i_start[i]:i_end[i]] = gb.A[i,:]
    xE[i_start[i]:i_end[i]] = gb.E[i,:]
    xA_all = xA_all + xA
    xE_all = xE_all + xE


xA_all = xA_all + noiseA  
xE_all = xE_all + noiseE

end_time = time.time()

print(end_time - start_time)

# Plot the spectrum of the data
print('df = ', df)
plt.figure()
plt.loglog(2.0*df*((xp.abs(xA_all)).get())**2)
plt.loglog(noisevals_A.get())
plt.savefig('spec_A.png')

