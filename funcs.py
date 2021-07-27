# Imports

import pandas as pd
import os
import datetime
import glob
from cfhcpy.surf_utils import split_cii
from tqdm import tqdm
from copy import deepcopy
import nibabel as nib
import numpy as np
import scipy.stats as st
import scipy.signal as sig
from scipy.io import wavfile
import pandas as pd
import cortex
import matplotlib.pyplot as plt
from scipy import stats
from copy import deepcopy
import scipy.signal as signal
from nistats.hemodynamic_models import spm_hrf, spm_time_derivative, spm_dispersion_derivative
import h5py
from tqdm import tqdm



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

    import nibabel as nib
    from scipy.signal import savgol_filter
    import numpy as np
    import os

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
    if y[0]<half and y[-1]<half:
        hmx=[lin_interp(x, y, zero_crossings_i[0], half),
                lin_interp(x, y, zero_crossings_i[1], half)]
        fwhm = hmx[1] - hmx[0]
        
    # Or if the function is above the HM at the start, just take the location of the the only zero crossing.
    elif y[0]>half:
        
        fwhm=x[zero_crossings_i[-1]]
    # Or if the function is above the HM at the end - go from the maximum value of X to the only zero crossing. 
    elif y[-1]>half:
        fwhm=np.max(x)-x[zero_crossings_i[-1]]
        
    else:
        return np.nan
    if plot:
        plt.plot(x,signs)
        plt.plot(x,y)
        #print(x[zero_crossings_i])
    return fwhm
    
    
            
# Spectrogram downsampler.


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


# Functions for interacting with HDF data.


def h5_make(path,s):
    
    """ Makes a hdf file at the given path"""
    
    fname=os.path.join(path,s+'.hdf5')
    
    if not os.path.isfile(fname):
        f = h5py.File(fname, "w")
        f.close()
    elif os.path.isfile(fname):
        print('filename already exists')
    return fname



def hdfshutter():
    """ Forces HDF files to shut"""
    
    import gc
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
            except:
                pass # Was already closed
                
    


def h5_dump2(h5,dat,attribute):
    """ Dumps numpy array to a hdf5 file.


    Parameters
    ----------
    h5 : Path to the H5 file.
    dat: data to save
    attribute : Name for the attribute.

    Returns
    -------

    """
    
    f = h5py.File(h5, "a")

    if type(dat)==str:

        dt = h5py.special_dtype(vlen=str)
        dset = f.create_dataset(attribute,(1,),dtype=dt)
        dset[0]=dat

    elif isinstance(dat, (int, float)):
        dset = f.create_dataset(attribute,(1,))
        dset[0]=dat
        
    elif isinstance(dat,np.ndarray):
        f[attribute]=dat
        
    f.close()

    return attribute

 
    
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

    

class HCP_subject():
    
    
    """ HCP_subject
    Class for interacting with and loading the HCP data.
    Requires a cfhcpy analysis base initialised on a subject.

    """
    
    
    def __init__(self,ab):
        
        # Copy all of the analysis base
        self.ab=deepcopy(ab)
        # Get the csv for the subject.
        self.csv_path=os.path.join('/tank/shared/2020/hcp_movie/sessionSummaryCSV_7T',f'{self.ab.subject}_all.csv')
        
        if os.path.isfile(self.csv_path):
        
            self.csv=pd.read_csv(self.csv_path)
            # Get the date string - use this to determine whhat version of the video they viewed.
            datstring=self.csv['Acquisition Time'][0]
            year,month,day=datstring.split('-')
            self.date=datetime.datetime(int(year),int(month),int(day))
            self.postrefdate=self.date>=datetime.datetime(2014,8,21)
        
        else:
            self.postrefdate = bool(int(input('Prompt : After ref date? 1=True or 0=False')))
        
        if self.postrefdate:
            self.vidprefix='movies/Post_20140821_version/'
            
        else:
            self.vidprefix='movies/Pre_20140821_version/'
        
        self.ab.preferred_movie_path=self.vidprefix
        
    def get_data_path(self,run):
        
        """ get_data_path
            Returns data file given a run.
        """
        
        wildcard = os.path.join(self.ab.subject_base_dir,self.ab.experiment_dict['data_file_wildcard'].format(experiment_id=self.ab.experiment_dict['wc_exp'],run=self.ab.experiment_dict['runs'][run]))
        dpath=glob.glob(wildcard)[0]
        return dpath

    def get_data_paths(self):
        
         """get_data_paths
            Returns all data files.
        """
            
        dpaths=[]
        for run in range(len(self.ab.experiment_dict['runs'])):
            dpaths.append(self.get_data_path(run))
        self.dpaths=dpaths
        
    def get_dm_paths(self):
        
        """get_data_paths
            Returns the paths for the wav files for producing the spectrograms.
        """
        
        # Gets the wav files for the movies.
        wvs = sorted(glob.glob(os.path.join(self.ab.experiment_base_dir,self.ab.preferred_movie_path, '*.wav')))
        self.wvs=wvs
                
        
    def prep_data(self):
        
        """prep_data
            Returns all paths.
        """
        
        
        self.get_data_paths()
        self.get_dm_paths()
    
    def read_run_data(self,dat,dtype):
        
        """read_run_data
            Loads in the cifti data.
        """
        
        
        l,r=load_and_split_surf(dat)
        data=np.hstack([l,r])
        
    
        if dtype=='main': #Subset so that we remove the test sequence.
            data=data[:-self.ab.experiment_dict['test_duration'],:]
        
        elif dtype=='test':
            data=data[-self.ab.experiment_dict['test_duration']:,:]
            
        elif dtype=='all':
            data=data
        
        return data

    
    def read_all_data(self,dtype):
        
        """read_all_data
            Loads in all the cifti data and concatenates it into a list.
        """
        
        print('Reading in data')
        concat_data=[]
        for run in tqdm(range(len(self.ab.experiment_dict['runs']))):
            concat_data.append(self.read_run_data(self.dpaths[run],dtype))
        self.all_data=concat_data
        
        
    def read_into_dm(self,wavs,run,dtype,normalise,filt,zaxis,nperseg=1024):
        
        """read_into_dm
            Reads in a wavfile and produces the normalized spectrogram for the prf design matrix.
        """
        
        
        sample_rate,samples=wavfile.read(wavs[run]) # Load the wav file
        nTRs=self.ab.experiment_dict['run_durations'][run]
        
        # Get the spectrogram.
        frequencies, times, spectrogram = sig.spectrogram(samples, sample_rate,nperseg=nperseg) # Make spectrogram
        
        # Take the mean of the spectrogram every second.
        design_matrix=dwn(spectrogram,times,nTRs)
        
        if filt:
            
        # Filter the design matrix with savgol filter.
            design_matrix=savgol_filter(design_matrix,3,0,210,1,'float32')
        
        if dtype=='main':
            design_matrix=design_matrix[:,:-self.ab.experiment_dict['test_duration']] # Subset to remove the test sequence
        
        elif dtype=='test':
            design_matrix=design_matrix[:,-self.ab.experiment_dict['test_duration']:]
        
        if normalise:
            
            # Zscore over time.
            
            design_matrix=design_matrix/np.std(design_matrix,axis=1)[...,np.newaxis]
            
        
        return design_matrix,frequencies
    
    
    
    def read_all_dms(self,dtype,zscore,filt,zaxis):
        
        """read_all_dms
            Reads all wav files for all runs and transforms them into the spectrograms.
        """
        
        print('Creating design matrices')
        concat_dms=[]
        frequencies=[]
        for run in tqdm(range(len(self.ab.experiment_dict['runs']))):
            concat_dms.append(self.read_into_dm(self.wvs,run,dtype,zscore,filt,zaxis)[0])
            frequencies.append(self.read_into_dm(self.wvs,run,dtype,zscore,filt,zaxis)[1])
            
        self.all_dms=concat_dms
        self.all_frequencies=frequencies
        self.frequencies=np.mean(self.all_frequencies,axis=0)
        
        
    def make_trainfolds(self):
        
        """make_trainfolds
            Organises the data and design matrices into leave one run out training and test folds.
        """
        
        print('making training and test folds')
        data_train,dm_train=[],[]
        data_test,dm_test=[],[]
        folds=[]
        for run in tqdm(range(len(self.ab.experiment_dict['runs']))):
            train_runs = list(range(len(self.ab.experiment_dict['runs']))) #Create a list of the runs
            train_runs.remove(run) # remove the ith run
            folds.append(train_runs) #This is our data for this run.
            data_train.append(np.concatenate([self.all_data[t] for t in train_runs], axis=0)) #Thus concatenate this. 
            data_test.append(self.all_data[run]) # And put the left out data in the corresponding place for the test set.
            dm_train.append(np.concatenate([self.all_dms[t] for t in train_runs], axis=1))
            dm_test.append(self.all_dms[run])

        self.data_train=data_train
        self.data_test=data_test
        self.dm_train=dm_train
        self.dm_test=dm_test
        self.folds=folds # Also save the fold indexes.
        
        
        
    def import_data(self,dtype,zscore=True,filt=True,zaxis=1):
        
        """import_data
            Reads all data and produces all design matrices.
        """
        self.read_all_data(dtype)
        self.read_all_dms(dtype,zscore,filt,zaxis)
        self.make_trainfolds()
        