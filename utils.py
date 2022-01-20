import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml

import datetime
import glob
from tqdm import tqdm
from copy import deepcopy
import random
import subprocess
import h5py

import cortex
import nibabel as nib
from scipy.signal import savgol_filter
from scipy.io import wavfile
from scipy import stats
import scipy.signal as signal
from nistats.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative

from prfpy.rf import gauss1D_cart


# General fmri timecourse/ fitting utilities.


def create_hrf(hrf_params=[1.0, 1.0, 0.0],TR=1):
    """
        
    construct single or multiple HRFs        
    Parameters
    ----------
    hrf_params : TYPE, optional
    DESCRIPTION. The default is [1.0, 1.0, 0.0].
    Returns
    -------
    hrf : ndarray
    the hrf.
    
    
    Adapted from prfpy
    """
        
    hrf = np.array([np.ones_like(hrf_params[1])*hrf_params[0] *spm_hrf(tr=TR,oversampling=1,time_length=40)[...,np.newaxis],
    hrf_params[1] *spm_time_derivative(tr=TR,oversampling=1,time_length=40)[...,np.newaxis],hrf_params[2] *
    spm_dispersion_derivative(tr=TR,oversampling=1,time_length=40)[...,np.newaxis]]).sum(axis=0)
    

    return hrf.T


def convolve(hrf,regressor):
    
    """
    Performs standard HRF convolution with a regressor.     
    Parameters
    ----------
    hrf : a HRF (i.e. created by create_hrf)
    Regressor: A fmri timecourse regressor
    Returns
    -------
    conv : The convolve regressor.
    """
    
    
    
    hrf_shape = np.ones(len(regressor.shape), dtype=np.int)
    hrf_shape[-1] = hrf.shape[-1]
    conv=signal.fftconvolve(regressor,hrf.reshape(hrf_shape), mode='full', axes=(-1))[..., :regressor.shape[-1]]
    return conv



def rsq_for_model(data, model_tcs, return_yhat=False):
    
    """
    Parameters
    ----------
    data : numpy.ndarray
        1D or 2D, containing single time-course or multiple
    model_tcs : numpy.ndarray
        1D, containing single model time-course
    Returns 
    -------
    rsq : float or numpy.ndarray
        within-set rsq for this model's GLM fit, for all voxels in the data
    yhat : numpy.ndarray
        1D or 2D, model time-course for all voxels in the data
        
    Courtesy of knapenlab.

    """
    dm = np.vstack([np.ones(data.shape[-1]), model_tcs]).T
    betas = np.linalg.lstsq(dm, data.T)[0]
    yhat = np.dot(dm, betas).T
    rsq = 1-(data-yhat).var(-1)/data.var(-1)
    if return_yhat:
        return rsq, yhat,betas
    else:
        return rsq
    
    
def savgol_filter(data, polyorder, deriv, window_length,tr,fmt):
    
    """ Applies a savitsky-golay filter to a nifti-file.

    Fits a savitsky-golay filter to a 4D fMRI nifti-file and subtracts the
    fitted data from the original data to effectively re
    low-frequency
    signals.

    Parameters
    ----------
    in_file : str
    Absolute path to nifti-file.
    polyorder : int (default: 3)
    Order of polynomials to use in filter.
    deriv : int (default: 0)
    Number of derivatives to use in filter.
    window_length : int (default: 120)
    Window length in seconds.

    Returns
    -------
    out_file : str
    Absolute path to filtered nifti-file.


    Courtesy of knapenlab.
    """



    dims = data.shape
    

    # TR must be in seconds
    if tr < 0.01:
        tr = np.round(tr * 1000, decimals=3)
    if tr > 20:
        tr = tr / 1000.0

    window = np.int(window_length / tr)
    
    # Window must be odd
    if window % 2 == 0:
        window += 1

    data = data.reshape((np.prod(data.shape[:-1]), data.shape[-1]))
    data_filt = savgol_filter(data, window_length=window, polyorder=polyorder,
                              deriv=deriv, axis=1, mode='nearest')

    data_filt = data - data_filt + data_filt.mean(axis=-1)[:, np.newaxis]
    data_filt = data_filt.reshape(dims)
    data_filt=data_filt.astype(fmt)

    return data_filt
    
    
    
# FWHM functions.

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


def fwhm(x, y,plot=False):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    
    # If the function ever reaches less than half its maximum.
    if y[0]<half and y[-1]<half and len(zero_crossings_i)>1:
        hmx=[lin_interp(x, y, zero_crossings_i[0], half),
                lin_interp(x, y, zero_crossings_i[1], half)]
        fwhm = hmx[1] - hmx[0]
        
    elif y[0]<half and y[-1]<half and len(zero_crossings_i)==1:
        fwhm=x[-1]-x[zero_crossings_i[-1]]    
    
    # Or if the function is above the HM at the start, just take the location of the the only zero crossing.
    elif y[0]>half:
        
        fwhm=x[zero_crossings_i[-1]]
        
    # Or if the function is above the HM at the end - go from the maximum value of X to the only zero crossing. 
    elif y[-1]>half:
        fwhm=np.max(x)-x[zero_crossings_i[-1]]
        
    else:
        return np.nan
    if plot:
        plt.plot(x[:200],signs[:200])
        plt.plot(x[:200],y[:200])
        #print(x[zero_crossings_i])
    return fwhm
    
    
def octaves(x, y,plot=False):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    
    # If the function ever reaches less than half its maximum.
    if y[0]<half and y[-1]<half and len(zero_crossings_i)>1:
        hmx=[lin_interp(x, y, zero_crossings_i[0], half),
                lin_interp(x, y, zero_crossings_i[1], half)]
        octave=np.log(hmx[1]/hmx[0])/np.log(2)
        
    elif y[0]<half and y[-1]<half and len(zero_crossings_i)==1:
        octave=np.log(x[-1]/x[zero_crossings_i[-1]])/np.log(2)
    # Or if the function is above the HM at the start, just take the location of the the only zero crossing.
    elif y[0]>half:        
        octave=np.log(x[zero_crossings_i[-1]]/np.min(x))/np.log(2)
    # Or if the function is above the HM at the end - go from the maximum value of X to the only zero crossing. 
    elif y[-1]>half:        
        octave=np.log(np.max(x)/x[zero_crossings_i[-1]])/np.log(2)
        
    else:
        return np.nan
    if plot:
        plt.plot(x,signs)
        plt.plot(x,y)    
    return octave
    
            
def convert_fwhm(mus,sigmas,exps,frequencies,mask,do_octaves=False):
    
    fwhma=np.empty(len(mus))
    fwhma[:] = np.nan
    # Linear frequencies.
    mus,sigmas,exps=np.array(mus[mask]),np.array(sigmas[mask]),np.array(exps[mask])
    log_frequencies=np.log(frequencies)
    
    fwhms=[]

    for i in tqdm(range(len(mus))):
        
        # Construct the gaussian - inflate the sigma by the exponent.
        y=gauss1D_cart(log_frequencies,mus[i],sigmas[i]/np.sqrt(exps[i]))
        # Determine the FWHM of the function.
        if do_octaves==False:
            fwhms.append(fwhm(frequencies,y))
        elif do_octaves==True:
            fwhms.append(octaves(frequencies,y))
        
    fwhma[mask]=np.array(fwhms)
    
    return fwhma


def dwn(spectrogram,times,TRs,TR=1):
    
    """ Downsamples a spectrogram via taking the mean every TR.

    Takes the mean within each TR. I couldn't find an existing function that did exactly this,
    so made one myself. 
    
    Parameters
    ----------
    spectrogram : A spectrogram as prepared by scipy.spectrogram (third argument)
    times: times (i.e. second output of scipy.spectrogram)
    TRs : Number of TRs
    TR: the TR
    -------
    """

    startval=0
    endval=TR

    reduce=[]
    for i in tqdm(range(TRs)):
        subset=spectrogram[:,np.logical_and(times<endval,times>startval)] # Get the data between start and end
        val=np.mean(subset,axis=1) # Take the mean.
        reduce.append(val) # Add to list
        startval=startval+TR # Add TR to the start and end
        endval=endval+TR
    rarray=np.array(reduce) # Put into array.
    return(rarray.T)


    
def load_and_split_surf(cifti):
    
    """ Function for loading and splitting cifti into left and right hemisphere surfaces.
    Parameters
    ----------
    cifti : Path to the cifti file
    
    Returns
    -------
    l, left hemisphere.
    r, right hemisphere.

    """
    
    idxs= h5py.File('/tank/hedger/DATA/HCP_temp/Resources/cifti_indices.hdf5', "r")
    lidxs=np.array(idxs['Left_indices'])
    ridxs=np.array(idxs['Right_indices'])
    idxs.close()
    
    datvol=nib.load(cifti)
    dat=np.asanyarray(datvol.dataobj)
    
    # Populate left and right hemisphere.
    l,r,=dat[:,lidxs],dat[:,ridxs]
    
    l[:,lidxs==-1]=np.zeros_like(l[:,lidxs==-1])
    r[:,ridxs==-1]=np.zeros_like(r[:,ridxs==-1])
    
    
    return l,r


def masked_vert(data,mask):
    vdat=np.repeat(0,mask.shape[-1]).astype('float32')
    vdat[mask==1]=np.array(data)
    return vdat


def read_mask():
    myframe=pd.read_csv('/tank/hedger/DATA/HCP_temp/OUTPUTS/CSVS/sframe.csv')
    mymask=np.array(myframe['mask']).astype(bool)
    mask=mymask
    return mask


def test_pred(tseries_raw,speechpred,nspeechpred,betas):
    yhat=betas[-1]+(speechpred*betas[0])+(nspeechpred*betas[1])
    rsq = 1-(yhat-tseries_raw).var(0)/tseries_raw.var(0)
    return yhat,rsq


def split_cii(fn, workbench_split_command, workbench_resample_command='', resample=True):
    wbc_c = workbench_split_command.format(cii=fn, cii_n=fn[:-4])
    subprocess.call(wbc_c, shell=True)

    if resample:
        for hemi in ['L', 'R']:
            this_cmd = workbench_resample_command.format(
                metric_in=fn[:-4] + '_' + hemi + '.gii',
                hemi=hemi,
                metric_out=fn[:-4] + '_fsaverage.' + hemi + '.gii')
            plist = shlex.split(this_cmd)
            subprocess.Popen(plist)