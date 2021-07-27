# Repository for auditory prf analysis of HCP movie-watching data.

This repository accompanies the preprint: **Naturalistic Audiovisual Stimulation Reveals the Topographic Organization of Human Auditory Cortex**

https://www.biorxiv.org/content/10.1101/2021.07.05.447566v1.full

### Information:

1. This repository does not re-invent the wheel, making use of existing routines where possible. It leverages the routines implemented in *prfpy* for model fitting. Prfpy is a package dedicated to fitting prf models: https://github.com/VU-Cog-Sci/prfpy 
2. Moreover it also leverages an existing package for interacting with the HCP data: https://github.com/tknapen/hcp_movie/tree/master/cfhcpy . This package was used to support the 2021 PNAS publication *Topographic connectivity reveals task-dependent retinotopic processing throughout the human brain*. The *analysisbase* class that is used to interface with the HCP data is modified to create a *HCP_subject* class that handles all the data imports. 
3. Various functions, including the HCP_subject class are defined in funcs.py 
4. Different stages of analysis are carried out by different notebooks, as detailed below:

### Notebooks

1. *Average_subs*: Creates the two across-subject folds reported in the paper.
2. *Fit_prfs* : Performs the prf fitting for each subject.
3. *Null_model*: Performs the model fitting for the 'sound on' null model reported in the methods and Supplementary Material.
4. *Speech_model*: Performs the fitting for the speech-selective model reported in the paper. 

### Model outputs.

We have prepared an independent Figshare repository for the final model parameters produced here and reported in the paper. This includes a pycortex subject database entry that can be used to generate the figures shown in the manuscript: https://figshare.com/projects/Data_from_Naturalistic_Stimulation_Reveals_the_Topographic_Organization_of_Human_Auditory_Cortex_/117288


