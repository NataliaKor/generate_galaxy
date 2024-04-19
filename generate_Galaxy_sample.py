# Based on the code form Valeriya https://gitlab.in2p3.fr/korol/observationally-driven-population-of-galactic-binaries

import matplotlib.pyplot as plt
import numpy as np
#import cupy as xp
import scipy.constants as sc
import scipy.interpolate 
from astropy import constants as const
from astropy import units as u
from pygaia.astrometry.coordinates import CoordinateTransformation, Transformations

from tqdm import tqdm

#from gbgpu.gbgpu import GBGPU
from gbgpu.utils.constants import *

from noisemodel import *

from matplotlib import colors
from matplotlib import ticker 
from matplotlib import rc 
rc('font',**{'family':'serif','serif':['TX Times']})
rc('text', usetex=True)
import corner

import time

start_time = time.time() 

'''
 Total number of GBs in LISA frequency band
'''
V=6e11 #pc^3 is the total volume of the disc obtained by integrating Eq.(7) of our paper
rhoWD = 4.49e-3 #pc^-3 #DWD local soace density from Hollands et al. (2018), arXiv:1805.12590
fDWD = 0.01 #fraction of DWD per single WD as has been derived in our paper 
N = int(V*rhoWD*fDWD)
print('N = ', N)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# THIS HAS TO BE REMOVED FOR DATA GENERATION
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Redefine N for testing
N = 5000000 #500000


'''
 Masses. 
 Assume that the heaviest mass is the first one: m1. It follows distribution of the single DWD.
 Based on the SDSS spectroscopic sample of $\sim1500$ white dwarfs, Kepler et al. (2015,  arXiv:1411.4149) derived a white dwarf mass function that follows a three-component Gaussian mixture with means $\mu = \{0.65, 0.57, 0.81\}$ M$_\odot$, standard deviations $\sigma = \{0.044, 0.097, 0.187\}$ M$_\odot$, and respective weights $w = \{0.81, 014, 0.05\}$.

Observations show that at the main-sequence stage, the secondary stars follow a mass-ratio distribution that is approximately flat, rather than the same mass distribution as the primary star. Therefore, in our (default) model we draw $m_2$ from a flat distribution between $0.15$ M$_\odot$, the minimum mass of observed extremely-low-mass (ELM) white dwarfs, and $m_1$. For a small number of cases in which we get $m_1 < 0.25$\,M$_\odot$ i.e. falling into the ELM category, we draw $m_2$ from the range $[0.2, 1.2]$ M$_\odot$ with equal probability. Note that in the later case the definition of $m_1$ and $m_2$ is swapped. However, since the gravitational wave radiation depends only on the combined chirp mass this does not affect our analysis.


'''
def normal(x, mu, sig):
    return 1. / (np.sqrt(2 * np.pi) * sig) * np.exp(-0.5 * np.square(x - mu) / np.square(sig))

def trunc_normal(x, mu, sig, bounds=None):
    if bounds is None: 
        bounds = (-np.inf, np.inf)

    norm = normal(x, mu, sig)
    norm[x < bounds[0]] = 0
    norm[x > bounds[1]] = 0

    return norm

def sample_trunc(n, mu, sig, bounds=None):
    """ Sample `n` points from truncated normal distribution """
    x = np.linspace(mu - 5. * sig, mu + 5. * sig, 10000)
    y = trunc_normal(x, mu, sig, bounds)
    y_cum = np.cumsum(y) / y.sum()
    yrand = np.random.rand(n)
    sample = np.interp(yrand, y_cum, x)

    return sample

def gen_masses_default(N):                                            
    weights = np.array([0.81, 0.14, 0.05])                    
    gauss1 = sample_trunc(int(N*weights[0]), 0.649, 0.04, (0.15, 1.4))
    gauss2 = sample_trunc(int(N*weights[1]), 0.57, 0.097, (0.15, 1.4))
    gauss3 = sample_trunc(int(N*weights[2]), 0.81, 0.187, (0.15, 1.4))
    M1 = np.concatenate([gauss1,gauss2,gauss3])
    M2 = np.random.uniform(0.15,M1) #min limit could be the lowest mass ELM for example
    M2[M1<0.25]=np.random.uniform(0.2,1.2) #
    return M1, M2  

m1,m2 = gen_masses_default(N)*u.Msun

# Chirp mass
Mchirp = (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)

'''
  For each DWD we generate random 3D positions in the Galactic disc according to
  $\rho(R,z) = \rho_0 e^{-R/H_{\rm R}} {\rm sech}^2{z/h_{\rm z}},$ 
'''

def random_position_disc(N):
    HR    = 2.5   # in kpc 
    R_sun = 8.1   #in kpc
    hz    = 0.3   #in kpc

    rad_to_deg=57.295779513

    R = np.power(-13. * np.log(np.random.uniform(0,1,N)), 20./31.)
    phi =  np.random.uniform(0,2*np.pi,N)
    X   = np.random.uniform(0,1,N)
    z   = hz * np.log(-np.sqrt(-(X - 1) * (X + 1)) / (X - 1))
    x   = R * np.cos(phi)
    y   = R * np.sin(phi)
  
    d_proj = np.sqrt(np.power(x, 2) + np.power(y-R_sun, 2))
    d  = np.sqrt(np.power(x, 2) + np.power(y-R_sun, 2) + np.power(z, 2))  
    l     = rad_to_deg * np.arctan2(x,(y - R_sun)) # * rad_to_deg
    b     = rad_to_deg * np.arctan2(z,d_proj) # * rad_to_deg

    l[y > R_sun] -= (x[y > R_sun]/np.abs(x[y > R_sun])) * 180.

    return l,b,d 

l,b,d = random_position_disc(N)
transf_GAL2ECL = CoordinateTransformation(Transformations.GAL2ECL)

l_rad = np.deg2rad(l)
b_rad = np.deg2rad(b)

lam, beta = transf_GAL2ECL.transform_sky_coordinates(l_rad, b_rad)

'''
Separation distribution.
'''
t0 = 10e9*u.year
alpha = -1.3
flisa = 5e4 ### the minimum frequency detectable by LISA
a_max = (((flisa*u.s/4/np.pi)**2*const.G*(m1+m2))**(1/3)).to(u.au).value
K = 256/5.*const.G**3/const.c**5*m1*m2*(m1+m2).value

# This is a faster implementation, that uses matrices
# in order to quickly evaluate with different values of a, m1, m2
def separation_distribution(a, alpha=-1.3, m1=0.6*u.Msun, m2=0.6*u.Msun):

    K = 256/5*const.G**3/const.c**5*m1*m2*(m1+m2)
    t0 = 10e9*u.year

    if len(K) > 1:
    # create matrices to avoid loops
        a, K = np.meshgrid(a, K)

    x = a/(K*t0)**(1/4)

    if alpha == 1:
        N = x**3 * np.log(1 + x**(-4))
    else:
        N = x**(4 + alpha)*( ( 1 + x**(-4) )**((alpha+1)/4) - 1 )

  # remove units
    N = N.to(u.dimensionless_unscaled)
    N = N.value

  # normalize to get the PDF
    if len(K) == 1:
        area = np.sum(N[:-1]*np.diff(a.value))
        pdf = N/area
    else:
        area = np.sum(N[:,:-1]*np.diff(a.value), 1)
        fun = lambda x, y : x/y
        pdf = np.apply_along_axis(fun, 0, N, area)
    return pdf

def randpdf(x, y, n=1, smooth_factor=1):
    if smooth_factor > 1:
        dx = np.min(np.diff(x))/smooth_factor
        xvals = np.arange(np.min(x), np.max(x), dx)
        y = np.interp(xvals, x, y)
        x = xvals
        y[np.where(y<0)[0]] = 0
  
    pdf = y/np.sum(y)
    cdf = np.cumsum(pdf)

  # remove non-unique elements
    cdf, idx = np.unique(cdf, return_index=True)
    x = x[idx]

    r = np.random.rand(n)
    rpdf = np.interp(r, cdf, x)

    return rpdf


a_min = 2e4*u.km
a_max = ((2e4*u.s/4/np.pi)**2*const.G*(m1+m2))**(1/3)
da = 0.01*u.Rsun
a_vec = np.arange(a_min.to(u.au).value, np.max(a_max.to(u.au).value), da.to(u.au).value)*u.au

pdf = separation_distribution(a_vec, alpha, m1, m2)

a = np.empty(N)
for i in tqdm(range(N)):
    a_rand = randpdf(a_vec, pdf[i,:], 1)
    a[i] = a_rand[0].value
f = 2./((4*np.pi**2*(a*u.au.to(u.m))**3)/(sc.G*(m1.to(u.kg).value+m2.to(u.kg).value)))**0.5 #convert separation into GW frequency

'''
Derive values of the 
Amp  -- amplitude
fdot -- frequency derivative
'''
fdot = (96./5.) * np.power(np.pi,8./3.) * np.power(((const.G * Mchirp * const.M_sun * u.kg)/ np.power(const.c, 3)), 5./3.) * np.power(f, 11./3.)

Amp = (2.* np.power((const.G * Mchirp * const.M_sun * u.kg), 5./3.) * np.power((np.pi * f), 2./3.))/(np.power(const.c, 4)* u.pc.to(u.m, d*1000.)) 

parameters = np.vstack([np.log10(Amp.value), f, np.log(fdot.value), beta, lam]).T
labels = ["Amplitude", "Frequesncy", "Frequency derivative", "Latitude", "Longitude"] 

'''
Corner plot of parameters
'''
fig = corner.corner(parameters,
                    labels = labels,  
                    color = 'red',
                    plot_datapoints=False,
                    fill_contours=True,
                    bins = 50,
                    levels=[0.68,0.954,0.997],
                    weights=np.ones(parameters.shape[0])/parameters.shape[0],
                    plot_density=True)
plt.savefig('parameters.png')

'''
Sample remaining parameters
'''
iota_cos = np.random.uniform(-1.0, 1.0, N)
iota = np.arccos(iota_cos)
phi0 = np.random.uniform(0.0, 2.0 * np.pi, N)  # np.pi
psi =  np.random.uniform( np.pi, 2.0 * np.pi, N)    # np.pi

'''
Generate wavforms for all parameters.
What is the largest batch of data that I can create?
'''
parameters_waveform = np.array([np.asarray(Amp), np.asarray(f), np.asarray(fdot), np.zeros(f.shape), phi0, iota, psi, np.asarray(lam), np.asarray(beta)])
print('parameters_waveform.shape = ', parameters_waveform.shape)
np.save('galaxy.npy', parameters_waveform)

end_time = time.time()

print(end_time - start_time)
