# Repository for auditory prf analysis of HCP movie-watching data.

This repository accompanies the preprint: **Naturalistic Audiovisual Stimulation Reveals the Topographic Organization of Human Auditory Cortex**

https://www.biorxiv.org/content/10.1101/2021.07.05.447566v1.full

### Information:

1. This repository does not re-invent the wheel, making use of existing routines where possible. It leverages the routines implemented in *prfpy* for model fitting. prfpy is a package dedicated to fitting prf models: https://github.com/VU-Cog-Sci/prfpy . 

2. *base.py* includes a HCP_subject class that handles data imports and prepares a subject's data for analysis.
3. *analysis.py* includes an Analysis class that performs the analysis, given a HCP_subject class.
4. *aggregate.py* includes various functions for aggregating outcomes across subjects.
5. *vis.py* includes functions and classes for plotting flatmaps.
6. *utils.py* includes various utilities.
7. The *config.yml* sets the paths and parameters for analysis and plotting. Almost nothing is hard-coded, therefore changing the parameters in this file should suffice to alter the analysis.

### Notebooks

1. *Average_subs*: Creates the two across-subject folds reported in the paper and saves them out.
2. *Fit_prfs* : Performs the model fitting. 
3. *Data handling*: Creates the final model outputs and plots the outcomes. These are returned in the *Figures* folder

### Model outputs.

We have prepared an independent Figshare repository for the final model parameters produced here and reported in the paper. This includes a pycortex subject database entry that can be used to generate the figures shown in the manuscript: https://figshare.com/projects/Data_from_Naturalistic_Stimulation_Reveals_the_Topographic_Organization_of_Human_Auditory_Cortex_/117288


