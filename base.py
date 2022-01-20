# Imports

from utils import *
from vis import *
from prfpy.fit import Iso1DGaussianFitter
from prfpy.model import Iso1DGaussianModel
from prfpy.stimulus import PRFStimulus1Dn
from prfpy.stimulus import PRFStimulus1D
from prfpy.model import CSS_Iso1DGaussianModel
from prfpy.fit import CSS_Iso1DGaussianFitter
from prfpy.timecourse import sgfilter_predictions


class HCP_subject():
    
    
    """ HCP_subject
    Class for interacting with and loading the HCP data.

    """
    
    
    def __init__(self,subject,experiment_id,yaml_file):
        
        self.subject = subject
        self.experiment_id = experiment_id 
        self.startup(yaml_file)
        
    def startup(self,yaml_file):
        
        """startup
        Reads in yaml file, internalises subject information.
        """
        
        self.yaml_file=yaml_file
        self._internalize_config_yaml()
        print('''Starting analysis of subject {sj} on {system} with settings \n{sd}'''.format(
            sj=self.subject, system=self.system_name, sd=json.dumps(self.system_dict, indent=1)))
        self._setup_paths_and_commands()
        self.internalize_config(self.y,'paths')
        self.internalize_config(self.y,'surfinfo')
        self.internalize_config(self.y,'preproc')
        self.check_date()
        self.check_is_agg_sub()
        self.get_structinfo()
        
    def check_date(self):
        
        """check_date
        Checks the date the subject performed the scan assigns path to relevant design matrix.
        """
        
        # Get the csv for the subject.
        self.info_csv_path=os.path.join(self.info_path,f'{self.subject}_all.csv')
        
        # If there is an info file for the subject
        if os.path.isfile(self.info_csv_path):
        
            self.info_csv=pd.read_csv(self.info_csv_path)
            # Get the date string - use this to determine what version of the video they viewed.
            datstring=self.info_csv['Acquisition Time'][0]
            year,month,day=datstring.split('-')
            self.date=datetime.datetime(int(year),int(month),int(day))
            self.postrefdate=self.date>=datetime.datetime(2014,8,21)
        
        else:
            self.postrefdate = bool(int(input('Prompt : After ref date? 1=True or 0=False')))
        
        if self.postrefdate:
            self.vidprefix=self.late_mov_path
        else:
            self.vidprefix=self.early_mov_path
        self.preferred_movie_path=self.vidprefix
        
    def check_is_agg_sub(self):
        
        """check_is_agg_sub
        If the subject is an aggregate subject, we move the subject base dir away from the HCP root folder
        to where the aggregate subjects have been created.
        """
        
        # Point to the location of the aggregate subjects.
        if self.subject in self.agg_subs:
            self.subject_base_dir=os.path.join(self.agg_path,self.subject)
        
        
    def _internalize_config_yaml(self):
        
        """internalize_config_yaml
        Needs to have subject and experiment set up before running.
        Parameters
        ----------
        yaml_file : [string]
            path to yaml file containing config info
        """
        with open(self.yaml_file, 'r') as f:
            self.y = yaml.safe_load(f)

        # first, determine which host are we working on
        self.system_name = self._determine_system(self.y['systems'])
        self.system_dict = self.y['systems'][self.system_name]
        self.experiment_dict = self.y['experiments'][self.experiment_id]
        self.analysis_dict = self.y['analysis']

        for key in self.analysis_dict.keys():
            setattr(self, key, self.analysis_dict[key])
            
            
    def _determine_system(self, system_dict):
        
        """ determines the system this analysis is being run on, using uname
        and information in the 'systems' section of the config.yml.
        Returns
        -------
        ps : [string]
            which system this analysis is being run on.
        """

        base_dir = None
        uname = str(os.uname())
        possible_systems = list(system_dict.keys())
        possible_unames = [system_dict[s]['identifier']
                           for s in possible_systems]
        for ps, pu in zip(possible_systems, possible_unames):
            if pu in uname:
                return ps
        if base_dir == None:
            print('cannot find base_dir in "{uname}", exiting'.format(
                uname=uname))
            sys.exit()
            
            
    def _setup_paths_and_commands(self):
        
        """setup_paths
        sets up all the paths and commands necessary given an experiment id and subject
        """
        
        # set base directory paths dependent on experiment and subject
        self.experiment_base_dir = self.system_dict['base_dir'].format(
            experiment=self.experiment_id)
        self.subject_base_dir = os.path.join(
            self.experiment_base_dir, 'subjects', self.subject)

               
    def internalize_config(self,y,subdict):
        
        """internalize_config
        internalises a specific subdictionary of a yaml
        """
        
        subdict = y[subdict]

        for key in subdict.keys():
            setattr(self, key, subdict[key])
        
        
    def get_structinfo(self):
        
        """ get_structinfo
            Returns the mean surface info across subjects.
        """
        
        if self.subject in self.agg_subs:
            surfinfos=np.array([np.load(os.path.join(self.av_dir,'{param}.npy'.format(param=param))).flatten() for param in self.surf_params])
            self.structframe=pd.DataFrame(surfinfos.T,columns=self.surf_params)
        else:
            print('getting surf info')
            self.structframe=self.get_all_surf_params()
        

    def get_surf_info(self,param):
        
        """ get_surf_info
            Returns a pycortex surface param for a subject.
        """
        
        surfinfo=np.load(os.path.join(self.pcx_dir,self.surfinfo_wildcard.format(sub=self.subject,param=param)),allow_pickle=True)
        param=np.concatenate([surfinfo['left'],surfinfo['right']])
        return param


    def get_all_surf_params(self):
        
        """ get_all_surf_params
            Returns all pycortex surface params for each subject.
        """
        
        surfinfos=np.array([self.get_surf_info(param) for param in self.surf_params])
        params=pd.DataFrame(surfinfos.T,columns=self.surf_params)
        return params 
        
        
    def get_data_path(self,run):
        
        """ get_data_path
            Returns data file given a run.
        """
        
        wildcard = os.path.join(self.subject_base_dir,self.experiment_dict['data_file_wildcard'].format(experiment_id=self.experiment_dict['wc_exp'],run=self.experiment_dict['runs'][run]))
        dpath=glob.glob(wildcard)[0]
        return dpath

    def get_data_paths(self):
        
        """get_data_paths
            Returns all data files.
        """
        
        dpaths=[]
        for run in range(len(self.experiment_dict['runs'])):
            dpaths.append(self.get_data_path(run))
        self.dpaths=dpaths
        
    def get_dm_paths(self):
        
        """get_dm_paths
            Returns the paths for the wav files for producing the spectrograms.
        """
        
        # Gets the wav files for the movies.
        wvs = sorted(glob.glob(os.path.join(self.experiment_base_dir,self.preferred_movie_path, '*.wav')))
        self.wvs=wvs
                
        
    def prep_data(self):
        
        """prep_data
            Returns all paths.
        """
        
        self.get_data_paths()
        self.get_dm_paths()
    
    def read_run_data(self,dat):
        
        """read_run_data
            Loads in the cifti data.
        """
        
        
        l,r=load_and_split_surf(dat)
        data=np.hstack([l,r])
        
    
        if self.dtype=='main': #Subset so that we remove the test sequence.
            data=data[:-self.experiment_dict['test_duration'],:]
        
        elif self.dtype=='test':
            data=data[-self.experiment_dict['test_duration']:,:]
            
        elif self.dtype=='all':
            data=data
        
        return data

    
    def read_all_data(self):
        
        """read_all_data
            Loads in all the cifti data and concatenates it into a list.
        """
        
        print('Reading in data')
        concat_data=[]
        for run in tqdm(range(len(self.experiment_dict['runs']))):
            concat_data.append(self.read_run_data(self.dpaths[run]))
        self.all_data=concat_data
        
        
    def read_into_dm(self,wavs,run):
        
        """read_into_dm
            Reads in a wavfile and produces the normalized spectrogram for the prf design matrix.
        """
        
        sample_rate,samples=wavfile.read(wavs[run]) # Load the wav file
        nTRs=self.experiment_dict['run_durations'][run]
        
        # Get the spectrogram.
        frequencies, times, spectrogram = sig.nalspectrogram(samples, sample_rate,nperseg=self.nperseg) # Make spectrogram
        
        # Take the mean of the spectrogram every second.
        design_matrix=dwn(spectrogram,times,nTRs)
        
        
        if self.dtype=='main':
            design_matrix=design_matrix[:,:-self.experiment_dict['test_duration']] # Subset to remove the test sequence
        
        elif self.dtype=='test':
            design_matrix=design_matrix[:,-self.experiment_dict['test_duration']:]
        
        if self.standardise:
            
            # Zscore over time.
            
            design_matrix=design_matrix/np.std(design_matrix,axis=self.zaxis)[...,np.newaxis]
            
        
        return design_matrix,frequencies
    
    
    
    def read_all_dms(self):
        
        """read_all_dms
            Reads all wav files for all runs and transforms them into the spectrograms.
        """
        
        print('Creating design matrices')
        concat_dms=[]
        frequencies=[]
        for run in tqdm(range(len(self.experiment_dict['runs']))):
            concat_dms.append(self.read_into_dm(self.wvs,run)[0])
            frequencies.append(self.read_into_dm(self.wvs,run)[1])
            
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
        for run in tqdm(range(len(self.experiment_dict['runs']))):
            train_runs = list(range(len(self.experiment_dict['runs']))) #Create a list of the runs
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
        self.all_data=None
        self.all_dms=None
        
    
    def speechsplit(self,mov,cut=True):
        
        """speechsplit
            Splits a design matrix into speech and nonspeech sequences.
        """
    
        mylist=sorted(os.listdir(self.captionpath)) # Path to the speech csvs.
    
        frame=pd.read_csv(os.path.join(self.captionpath,mylist[mov]),delimiter=';')
    
        frame['Startsecs']=frame['Start time in milliseconds']/1000 # Convert to seconds
        frame['Endsecs']=frame['End time in milliseconds']/1000
    
        starts=np.array(frame['Startsecs'].astype(int)) 
        ends=np.array(frame['Endsecs'].astype(int))
        event=np.zeros(self.experiment_dict['run_durations'][mov])
    
        # Loop through the instances of speech as defined in the CSV 
        for i in range(starts.shape[0]):
            event[starts[i]:ends[i]]=1 # Code instances of speech as 1.
        if self.dtype=='main':
            event=event[:-self.experiment_dict['test_duration']]
        other=1-event
        other[self.dm_test[mov][0]==0]=0
        
        return event,other
    
    
    def make_speech_dms(self):
        
        """make_speech_dms
           makes all the speech and nonspeech design matrices.
        """
    
        speechdms_train,nspeechdms_train=[],[]
        speechdms_test,nspeechdms_test=[],[]
        speechdat,nspeechdat=[],[]

        # Go through each fold
        for i in range(len(self.dm_train)):

            # Make the design martices for each fold train and test. 
            dmtrain=np.hstack([self.speechsplit(t) for t in self.folds[i]])
            dmtest=self.speechsplit(i)
  
            tempsp_train,tempnsp_train=np.copy(self.dm_train[i]),np.copy(self.dm_train[i])
            tempsp_test,tempnsp_test=np.copy(self.dm_test[i]),np.copy(self.dm_test[i])
        
            # Silence needs to remain silence
            tempsp_train[:,dmtrain[0]==0]=0  
            tempnsp_train[:,dmtrain[1]==0]=0
            tempsp_test[:,dmtest[0]==0]=0
            tempnsp_test[:,dmtest[1]==0]=0
    
            speechdms_train.append(tempsp_train)
            nspeechdms_train.append(tempnsp_train)
    
            speechdms_test.append(tempsp_test)
            nspeechdms_test.append(tempnsp_test)
            
            speechdat.append(self.data_train[i][np.where(np.mean(speechdms_train[i],axis=0)!=0)[0],:])
            nspeechdat.append(self.data_train[i][np.where(np.mean(nspeechdms_train[i],axis=0)!=0)[0],:])
        
        self.speechdms_train=speechdms_train
        self.nspeechdms_train=nspeechdms_train
        self.speechdms_test=speechdms_test
        self.nspeechdms_test=nspeechdms_test
        self.speechdat=speechdat
        self.nspeechdat=nspeechdat
            

    def import_data(self):
        
        """import_data
            Reads all data and produces all design matrices.
        """
        
        self.read_all_data()
        self.read_all_dms()
        self.make_trainfolds()
        
        
    def prepare_out_dirs(self,analysis_name):
        
        """prepare_out_dirs
            Prepares all output directories for a subject.
        """
        
        self.analysis_name=analysis_name
        
        self.out_csv=os.path.join(self.out_base,self.analysis_name,'csvs',self.subject)
        self.out_flat=os.path.join(self.out_base,self.analysis_name,'flatmaps',self.subject)
        self.out_webGL=os.path.join(self.out_base,self.analysis_name,'webGL',self.webGL_wildcard.format(Subject=self.subject))

        if not os.path.isdir(self.out_flat):
            os.makedirs(self.out_flat,exist_ok = True)

        if not os.path.isdir(self.out_csv):

            os.makedirs(self.out_csv,exist_ok = True)
            
        if not os.path.isdir(self.out_webGL):

            os.makedirs(self.out_webGL,exist_ok = True)
            
        self.avcsv=os.path.join(self.out_csv,self.csv_wildcard.format(Subject=self.subject))
        self.longcsv=os.path.join(self.out_csv,self.long_csv_wildcard.format(Subject=self.subject))
        
    



            
            
            
        
        
        

    
        
        

    
        
    
        