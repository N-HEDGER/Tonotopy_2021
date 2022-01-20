from utils import *
from base import *

class Aggregator():
    
    """ aggregator
       Performs a set of functions to aggregate outputs of individual fits.

    """
    
    def __init__(self,base_dir,yaml):
        self.yaml=yaml
        self._internalize_config_yaml()
        
        self.base_dir=base_dir
        self.csv_dir=os.path.join(self.base_dir,'csvs')
        self.flist=os.listdir(self.csv_dir)
        self.filtered_flist=[x for x in self.flist if not any(y in x for y in self.invalid_subjects)]
        
        # Get all the individual subject fits.
        self.all_subs=[str(x) for x in self.y['analysis']['full_data_subjects']]
        self.incomplete_subs=list(set(self.all_subs).difference(self.filtered_flist))
        
        self.out_dir=os.path.join(self.base_dir,'aggregates')
        self.out_csv=os.path.join(self.out_dir,'csvs')
        self.out_flat=os.path.join(self.out_dir,'flatmaps')
        
        os.makedirs(self.out_csv,exist_ok = True)
        os.makedirs(self.out_flat,exist_ok = True)
        
        self.prf_dict = self.y['auditory_prf']
        self.folds=range(1,self.nfolds+1)
        
    def _internalize_config_yaml(self):
        
        """internalize_config_yaml

        """
        
        with open(self.yaml, 'r') as f:
            self.y = yaml.safe_load(f)

        self.agg_dict = self.y['agg']

        for key in self.agg_dict.keys():
            setattr(self, key, self.agg_dict[key])
    
        
    def read_all_subjects(self):
        
        """ read_all_subjects
        Read all subjects and return them into a frame.
        
        """
        
        self.csvs=[pd.read_csv(os.path.join(self.csv_dir,self.csv_wildcard.format(subno=f))) for f in tqdm(self.filtered_flist)]
        self.all_frames = pd.concat(self.csvs) # Read all the frames into a concatanated dataframe.
    
    
    def read_av_subjects(self):
        
        """ read_av_subjects
        Reads the averaged subjects (early and late)
        
        """
        
        self.avcsvs=[pd.read_csv(os.path.join(self.csv_dir,self.csv_wildcard.format(subno=f))) for f in tqdm(self.av_subs)]
        self.av_frames = pd.concat(self.avcsvs) 
        
        
    def read_long_av_subjects(self):
        
        """ read_long_av_subjects
        Reads the averaged subjects (early and late) (not collapsed across folds) 
        
        """
        
        self.long_avcsvs=[pd.read_csv(os.path.join(self.csv_dir,self.long_csv_wildcard.format(subno=f))) for f in tqdm(self.av_subs)]
        self.long_av_frames = pd.concat(self.long_avcsvs)
        
        
        self.long_av_frames.index=np.tile(range(self.nverts*self.nfolds),2) # Alter the index so subsequent averaging is across the subjects.
        
    
    def make_long_grand_average(self):
        
        """ make_long_grand_average
        Average the long data across subjects.
        
        """
        
        self.long_grand_frame=self.summarise_fits(self.long_av_frames,self.vars2avagg,self.vars2wavaggfold) # Average across subjects.
        self.long_grand_frame['fold']=np.repeat(self.folds,self.nverts) # Create variable to indicate folds.
    
    def make_fold_splits(self):
        """ make_fold_splits
        make fold-wise splits of the data
        
        """
        
        self.combs=list(itertools.combinations(self.folds, 2)) # Get all pairwise combinations of folds.
        # Make list of dataframes that contain these folds.
        self.combframes=[self.long_grand_frame[self.long_grand_frame['fold'].isin(comb)] for comb in self.combs] 
        for comb in self.combframes:
            comb.index=np.tile(range(self.nverts),2) # Now alter the index so we average across the pairs of folds.
        self.foldsplits=[self.summarise_fits(comb,self.vars2avagg,self.vars2wavaggfold) for comb in self.combframes] # Perform the averaging.

        
    def make_conversions(self,mask,frequencies):
        """ make_conversions
        Make the necessary conversions to the fold splits.
        
        """
        
        for fsplit in self.foldsplits:
            
            fsplit['fwhm']=convert_fwhm(fsplit['mu'],fsplit['sigma'],fsplit['n'],frequencies,mask=mask)
            fsplit['mu']=np.exp(fsplit['mu'])
        
    def export_fold_splits(self): 
        """ export_fold_splits
        Exports the fold splits
        
        """
        
        for ind, foldsplit in enumerate(self.foldsplits):
            # Saveout the foldsplits.
            foldsplit.to_csv(os.path.join(self.out_csv,self.foldsplit_wildcard.format(comb='_'.join([str(i) for i in self.combs[ind]]))))

    
    def make_grand_average(self):
        """ make_grand_average
        averages the aggregated subjects.
        
        """
        
        self.grand_frame=self.summarise_fits(self.av_frames,self.vars2avagg,self.vars2wavagg)
        
        
    
    def weighted_mean(self,df, values, weights, groupby):
        df = df.copy()
        grouped = df.groupby(groupby)
        df['weighted_average'] = df[values] / grouped[weights].transform('sum') * df[weights]
        return grouped['weighted_average'].sum(min_count=1) #min_count is required for Grouper objects
    
    
    def summarise_fits(self,lframe,vars2av,vars2wav):

        """ Summarise_fits

        Performs averaging, or weighted averaging to make a summary dataframe across folds.

        ----------

        """
        lframe=lframe.drop(['grouper'], axis=1,errors='ignore')
        
        lframe['grouper']=lframe.index
        
        xval4weights=np.copy(lframe['xval_R2'])
        xval4weights[xval4weights<0]=0.00001
        lframe['weights']=xval4weights
        
        vars2av=vars2av
        vars2wav=vars2wav
        
        wavs=pd.DataFrame(np.array([self.weighted_mean(lframe,var,'weights','grouper') for var in vars2wav]).T)
        wavs.columns=vars2wav

        # Regular averaging of variance explained.
        avs=pd.concat([lframe.groupby('grouper', as_index=False)[var].mean() for var in vars2av],axis=1)
        
        av_frame=pd.concat([wavs,avs],axis=1)
        
        return av_frame
    
    def make_av_frame(self):
        
        """ make_av_frame
        averages the individual subjects.
        
        """
        
        self.av_frame=self.summarise_fits(self.all_frames,self.vars2avagg,self.vars2wavagg)
        
    def sort_subjects(self):
        
        """ sort_subjects
        Sorts the individual subject fits by median variance explained.
        
        """
        self.fitvec=[np.nanmedian(csv['xval_R2']) for csv in self.csvs]
        # Get order.
        self.ords=np.argsort(self.fitvec)
        
    def get_botmidtop(self):
        """ get_botmidtop
        Gets the bottom, middle and top 5 subjects.
        
        """
        self.sort_subjects()
        self.midlist=int(len(self.csvs)/2)
        self.botcsvs=[self.csvs[csv] for csv in self.ords[:5]]
        self.topcsvs=[self.csvs[csv] for csv in self.ords[-5:]]
        self.midcsvs=[self.csvs[csv] for csv in self.ords[self.midlist-2:self.midlist+3]]
        
        self.botframes=pd.concat(self.botcsvs)
        self.topframes=pd.concat(self.topcsvs)
        self.midframes=pd.concat(self.midcsvs)
        
    def botmidtop_avs(self):
        
        """ botmidtop_avs
        Makes averages from the bottom mid and top 5 subjects.
        
        """
        
        self.get_botmidtop()
        self.botav=self.summarise_fits(self.botframes,self.vars2avagg,self.vars2wavagg)
        self.midav=self.summarise_fits(self.midframes,self.vars2avagg,self.vars2wavagg)
        self.topav=self.summarise_fits(self.topframes,self.vars2avagg,self.vars2wavagg)
    
    
    def export_grandav(self):
        
        """ export_grandav
        Exports the grand average subject.
        
        """
        
        self.grand_frame.to_csv(os.path.join(self.out_csv,'grand_average.csv'))
    
    def export_avs(self):
        
        """ export_avs
        Exports the bottom, middle and top 5 subjects and associated averages.
        
        """
        
        self.botav.to_csv(os.path.join(self.out_csv,'bot_average.csv'))
        self.midav.to_csv(os.path.join(self.out_csv,'mid_average.csv'))
        self.topav.to_csv(os.path.join(self.out_csv,'top_average.csv'))
        
        for i in range(5):
            self.botcsvs[i].to_csv(os.path.join(self.out_csv,self.bot_wildcard.format(participant=str(i))))
            self.topcsvs[i].to_csv(os.path.join(self.out_csv,self.top_wildcard.format(participant=str(i))))
            self.midcsvs[i].to_csv(os.path.join(self.out_csv,self.mid_wildcard.format(participant=str(i))))