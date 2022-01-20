from utils import *
from base import *


class Analysis:
    
    """ Analysis
       Performs analyses on the HCP_subject.

    """
    
    def __init__(self,subject):
        
        # Copy all of the analysis base
        self.sub=subject
        self.setup_mapping()
    
    def setup_mapping(self):
        
        """setup_mapping
            Sets up the frequency mapping for the fitting.
        """
        
        # Internatlize the analysis dictionary
        self.internalize_config(self.sub.y,'auditory_prf')
        
        if self.log=='natural':
            self.frequencies=np.log(self.sub.frequencies) 

        elif self.log=='log10':
            self.frequencies=np.log10(self.sub.frequencies)

        elif self.log=='nolog':
            self.frequencies=sub.frequencies
            
        if not hasattr(self,'mask'):
            self.mask=np.ones(self.sub.nverts)
            
    def internalize_config(self,y,subdict):
        
        subdict = y[subdict]

        for key in subdict.keys():
            setattr(self, key, subdict[key])
        
    def fit(self,fold):
        
        """fit
            Fits the model for a given fold.
        """
        
        
        null=self.fit_null_model(fold)
        print('fit null model')
        
        # Create a vector for the task length to allow filtering of each movie.
        blengths=np.array(self.sub.experiment_dict['run_durations'])[np.array(self.sub.folds[fold])]-self.sub.experiment_dict['test_duration']
        tl=list(blengths)
                
        blockarr=np.repeat(self.sub.folds[fold],blengths)
        
        train_stim=PRFStimulus1D(self.sub.dm_train[fold],self.frequencies,TR=self.TR,task_lengths=tl)

        test_stim=PRFStimulus1D(self.sub.dm_test[fold],self.frequencies,TR=self.TR)

        # Make model using the train stimulus.
        mymod=Iso1DGaussianModel(train_stim,normalise_RFs=self.normalise_RFs,filter_predictions=self.filter_predictions,filter_type=self.filter_type,filter_params=self.fparams)

        mymod.create_grid_predictions(self.mu_min,self.mu_max,self.mu_steps,self.sigma_min,self.sigma_max,self.sigma_steps,self.func)

        # Initial Gaussian pRF fitting.
        print('grid fitting')
        gf = Iso1DGaussianFitter(data=self.sub.data_train[fold].T,model=mymod, n_jobs=self.njobs)
        gf.grid_fit(self.mu_min,self.mu_max,self.mu_steps,self.sigma_min,self.sigma_max,self.sigma_steps,self.func,verbose=True,n_batches=self.n_batches)

        # Do iterative fitting with gauss pRF.
        print('iterative fitting')
        bounds=((self.mu_min,self.mu_max),(self.sigma_min,self.sigma_max),(None,None),(None,None))
        gf.rsq_mask=self.mask.astype(bool)
        gf.iterative_fit(rsq_threshold=0,verbose=True,bounds=bounds)


        # Do cross validation on gauss pRF.
        print('cross validation')
        gf.xval(test_data=self.sub.data_test[fold].T,test_stimulus=test_stim)

        # CSS fiting - define CSS model.
        print('CSS fitting')
        CSSmod=CSS_Iso1DGaussianModel(train_stim,normalise_RFs=self.normalise_RFs,filter_predictions=self.filter_predictions,filter_type=self.filter_type,filter_params=self.fparams)
        CSSmod.func=self.func

        # Define a CSS model that takes the gaussian params from the previous fitter as starting values.
        CSS_extender=CSS_Iso1DGaussianFitter(model=CSSmod,data=self.sub.data_train[fold].T, n_jobs=self.njobs, fit_hrf=False,previous_gaussian_fitter=gf)
        
        
        if self.constrain_n==False:
            CSSbounds=((self.mu_min,self.mu_max),(self.sigma_min,self.sigma_max),(None,None),(None,None),(self.exp_min,self.exp_max))
        elif self.constrain_n==True:
            print('constraining n parameter!')
            CSSbounds=((self.mu_min,self.mu_max),(self.sigma_min,self.sigma_max),(None,None),(None,None),(self.exp_const-1e-6,self.exp_const+1e-6))
            CSSbounds=np.array(CSSbounds)
            CSSbounds=np.repeat(CSSbounds[np.newaxis,...], gf.gridsearch_params.shape[0], axis=0)
            
        
        CSS_extender.rsq_mask=self.mask.astype(bool)

        # Do the iterative fitting. Just fit everything. Don't threshold based on R2. 
        CSS_extender.iterative_fit(rsq_threshold=0,verbose=True,bounds=CSSbounds)

        # CSS cross-validation
        print('CSS cross validation')
        CSS_extender.xval(test_data=self.sub.data_test[fold].T,test_stimulus=test_stim)
            
        foldframe=pd.DataFrame(CSS_extender.iterative_search_params,columns=self.fit_columns)
        foldframe['xval_R2']=CSS_extender.CV_R2
        
        # Numpy is not good with dealing with negaive values for weighted mean. Therefore assign a tiny value.
        xval4weights=np.copy(CSS_extender.CV_R2)
        xval4weights[xval4weights<0]=0.00001
        foldframe['weights']=xval4weights
        foldframe['null']=null
        
        return foldframe
    
    def fit_with_starting_params(self,fold):
        
        """fit_with_starting_params
            Fits the model for a given fold (with CSS starting params)
        """
        
        null=self.fit_null_model(fold)
        
        blengths=np.array(self.sub.experiment_dict['run_durations'])[np.array(self.sub.folds[fold])]-self.sub.experiment_dict['test_duration']
        blockarr=np.repeat(self.sub.folds[fold],blengths)
        task_lengths=list(blengths)
        
        train_stim=PRFStimulus1D(self.sub.dm_train[fold],self.frequencies,TR=self.TR,task_lengths=task_lengths)

        # Create test stimulus.
        test_stim=PRFStimulus1D(self.sub.dm_test[fold],self.frequencies,TR=self.TR)
        
        # CSS fiting.
        print('CSS fitting')
        CSSmod=CSS_Iso1DGaussianModel(train_stim,normalise_RFs=self.normalise_RFs,filter_predictions=self.filter_predictions,filter_type=self.filter_type,filter_params=self.fparams)
        CSSmod.func=self.func
        
        CSS_extender=CSS_Iso1DGaussianFitter(model=CSSmod,data=self.sub.data_train[fold].T, n_jobs=self.njobs, fit_hrf=self.fit_hrf)
        
        bounds=((self.mu_min,self.mu_max),(self.sigma_min,self.sigma_max),(None,None),(None,None),(self.exp_min,self.exp_max))
        
        # Define starting params.
        
        CSS_extender.rsq_mask=self.mask.astype(bool)
        CSS_extender.iterative_fit(rsq_threshold=0,verbose=True,bounds=bounds,starting_params=self.starting_params)
        
        print('CSS cross validation')
        CSS_extender.xval(test_data=self.sub.data_test[fold].T,test_stimulus=test_stim)
        
        foldframe=pd.DataFrame(CSS_extender.iterative_search_params,columns=self.fit_columns)
        foldframe['xval_R2']=CSS_extender.CV_R2
        
        xval4weights=np.copy(CSS_extender.CV_R2)
        xval4weights[xval4weights<0]=0.00001
        foldframe['weights']=xval4weights
        foldframe['null']=null
        
        return foldframe
    
    
        
    def fit_all_folds(self):

        """ fit_all_folds

        Performs fitting on all folds.

        ----------

        """
        
        self.CSS_fitted_params=[] # Make list to populate with CSS params.
        self.CSS_xval_R2=[] # Make list to populate with CSS xval performance.

        if hasattr(self,'starting_params'):
            self.all_fits=[self.fit_with_starting_params(i) for i in tqdm(range(len(self.sub.dm_train)))]
            
        else:
            self.all_fits=[self.fit(i) for i in tqdm(range(len(self.sub.dm_train)))]
            
        self.fits_concat = pd.concat(self.all_fits)
        self.fits_concat['grouper']=self.fits_concat.index # Averaging takes place across vertex number.
        self.fits_concat['fold']=np.repeat(range(len(self.sub.dm_train)),self.sub.nverts)
                    
        
    def fit_null_model(self,fold):
        
        """fit_null_model
            Fits the null model to the data.
        """
        
        hrf=create_hrf()
        empty=np.zeros(self.sub.dm_test[fold][0].shape[0]) 
        empty[np.where(self.sub.dm_test[fold][0]!=0)]=1 # Create a regressor where any sounds are coded 
        empty=empty.reshape(1,empty.shape[0]) 
        tc=convolve(hrf,empty) # Convolve with hrf
        f_tc=sgfilter_predictions(tc,window_length=self.fparams['window_length'], polyorder=self.fparams['polyorder'])
        data=self.sub.data_test[fold] # Data to fit.
        null_r2=rsq_for_model(data.T,f_tc)
        null_r2=np.array(null_r2)
        return null_r2
    
    
    
    def fit_speech_model(self,fold):
        
        """fit_speech_model
            Takes the CSS model predictions and uses them to perform speech-selective model fitting
        """

        indices=np.where(self.mask==True)[0]

        blengths=np.array(self.sub.experiment_dict['run_durations'])[np.array(self.sub.folds[fold])]-self.sub.experiment_dict['test_duration']
        blockarr=np.repeat(self.sub.folds[fold],blengths)
        
        # Design matrices for speech and nonspeech, training and test.
        speechstim_train=PRFStimulus1D(self.sub.speechdms_train[fold],self.frequencies,TR=self.TR,block_inds=blockarr)
        nspeechstim_train=PRFStimulus1D(self.sub.nspeechdms_train[fold],self.frequencies,TR=self.TR,block_inds=blockarr)
        speechstim_test=PRFStimulus1D(self.sub.speechdms_test[fold],self.frequencies,TR=self.TR)
        nspeechstim_test=PRFStimulus1D(self.sub.nspeechdms_test[fold],self.frequencies,TR=self.TR)


        # Make corresponding prf models.
        speechmod_train=CSS_Iso1DGaussianModel(speechstim_train,normalise_RFs=self.normalise_RFs,filter_predictions=self.filter_predictions,filter_type=self.filter_type,filter_params=self.fparams)
        speechmod_train.func=self.func

        nspeechmod_train=CSS_Iso1DGaussianModel(nspeechstim_train,normalise_RFs=self.normalise_RFs,filter_predictions=self.filter_predictions,filter_type=self.filter_type,filter_params=self.fparams)
        nspeechmod_train.func=self.func

        speechmod_test=CSS_Iso1DGaussianModel(speechstim_test,normalise_RFs=self.normalise_RFs,filter_predictions=self.filter_predictions,filter_type=self.filter_type,filter_params=self.fparams)
        speechmod_test.func=self.func

        nspeechmod_test=CSS_Iso1DGaussianModel(nspeechstim_test,normalise_RFs=self.normalise_RFs,filter_predictions=self.filter_predictions,filter_type=self.filter_type,filter_params=self.fparams)
        nspeechmod_test.func=self.func

    # Make the predictions based on CSS model for each of these design matrices.
        speechpreds=[speechmod_train.return_prediction(*list(np.array(self.fits_concat[self.fits_concat['fold']==fold].iloc[index][:5])))[0] for index in indices]
        nspeechpreds=[nspeechmod_train.return_prediction(*list(np.array(self.fits_concat[self.fits_concat['fold']==fold].iloc[index][:5])))[0] for index in indices]

        speechpreds_test=[speechmod_test.return_prediction(*list(np.array(self.fits_concat[self.fits_concat['fold']==fold].iloc[index][:5])))[0] for index in indices]
        nspeechpreds_test=[nspeechmod_test.return_prediction(*list(np.array(self.fits_concat[self.fits_concat['fold']==fold].iloc[index][:5])))[0] for index in indices]


        yhats,betass,rsqs=[],[],[]


        # Now do some fitting
        
        i=0
        for index in tqdm(indices):

            # Make design matrix (speech prediction, nspeech prediction, intercept).
            dm=np.vstack([speechpreds[i],nspeechpreds[i],np.repeat(1,nspeechpreds[i].shape)])

            # Get the data for this index.
            tseries_raw=self.sub.data_train[fold][:,index]
            tseries_raw=np.nan_to_num(tseries_raw)
            dm=np.nan_to_num(dm)

            # Solve the regression equation.
            betas, _, _, _ = np.linalg.lstsq(dm.T, tseries_raw.T)
            yhat = np.dot(betas.T, dm)
            rsq = 1-(yhat-tseries_raw).var(0)/tseries_raw.var(0)

            # Save the betas, rsq
            rsqs.append(rsq)
            yhats.append(yhat)
            betass.append(betas)
            i=i+1

        # Make into arrays the same shape as the full dataset. 
        R2=masked_vert(np.array(rsqs),self.mask)
        betas=np.array(betass)
        betdiff=masked_vert(betas[:,0]-betas[:,1],self.mask)

        # Test the predictions on the left out data.
        i=0
        perf=[]
        for index in tqdm(indices):

            # We derive the out of sample predicitons from the linear combination of speech and nonspeech predictions
            # that we just estimated.

            # We test such predictions on the left-out data to apply the same cross-validation strategy as for the CSS model.
            res=test_pred(self.sub.data_test[fold][:,index],speechpreds_test[i],nspeechpreds_test[i],betass[i])

            perf.append(res[1])
            i=i+1

        xval=masked_vert(np.array(perf),self.mask)
        sp_frame=pd.DataFrame(np.array([R2,xval,betdiff]).T,columns=self.speech_fit_columns)

        return sp_frame
    
    def fit_speech_models(self):
        
        """fit_speech_models
            Fits all the speech models. Appends the resulting frame to the CSS outcomes.
        """
    
        self.all_speech_fits=[self.fit_speech_model(i) for i in tqdm(range(len(self.sub.dm_train)))]
        self.speech_fits_concat = pd.concat(self.all_speech_fits)
        self.fits_concat=pd.concat([self.fits_concat,self.speech_fits_concat],axis=1)
        
        
    def summarise_fits(self):

        """ Summarise_fits

        Performs averaging, or weighted averaging to make a summary dataframe across folds.

        ----------

        """

        
        self.wavs=pd.DataFrame(np.array([self.weighted_mean(self.fits_concat,var,'weights','grouper') for var in self.vars2wav]).T)
        self.wavs.columns=self.vars2wav

        # Regular averaging of variance explained.
        self.avs=pd.concat([self.fits_concat.groupby('grouper', as_index=False)[var].mean() for var in self.vars2av],axis=1)
        
        self.av_frame=pd.concat([self.wavs,self.avs],axis=1)
        
        
    def weighted_mean(self,df, values, weights, groupby):
        
        """ weighted_mean

        Performs weighted averaging on a pandas dataframe.

        ----------

        """
        
        df = df.copy()
        grouped = df.groupby(groupby)
        df['weighted_average'] = df[values] / grouped[weights].transform('sum') * df[weights]

        return grouped['weighted_average'].sum(min_count=1) #min_count is required for Grouper objects
    
    def finalize_frame(self):
        
        """ finalize_frame
        Adds some extra details to the final dataframe
        ----------

        """
    
        self.make_conversions()
        self.bundle_surf_params()
        
    
    def make_conversions(self):
        
        """ make_conversions
        Converts mu into hz.
        Converts sigma to FWHM.
        Makes replaces 0s in the R2 with np.nan (useful for plotting.)
        ----------

        """
        
        
        self.av_frame['fwhm']=convert_fwhm(self.av_frame['mu'],self.av_frame['sigma'],self.av_frame['n'],self.sub.frequencies,mask=self.mask)
        
        self.av_frame['mu']=np.exp(self.av_frame['mu'])
        
        self.av_frame.loc[(1-(self.mask.astype(int))).astype(bool), 'R2'] = np.nan
        self.av_frame.loc[(1-(self.mask.astype(int))).astype(bool), 'xval_R2'] = np.nan
    
    
    def bundle_surf_params(self):
        
        """ bundle_surf_params
        Adds the surface parameters for the subject to the output frame.
        ----------

        """
        
        self.av_frame=pd.concat([self.av_frame,self.sub.structframe],axis=1)
    
        
    def saveout(self):

        """ saveout
        Saves out the CSV file.

        """
        
        self.finalize_frame()
       
        
        self.fits_concat.to_csv(self.sub.longcsv)


        # Also save to a common directory.
        self.av_frame.to_csv(self.sub.avcsv)