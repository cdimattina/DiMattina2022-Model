README.txt

OVERVIEW
--------
This is the README file for the public code to train the models in the paper
DiMattina et al. (2022). "Distinguishing shadows from surface boundaries using
local achromatic cues". PLoS Computational Biology (In Press). Please see the
paper for details and background.

The models are trained to distinguish the shadow set (SHAD) and one of two occlusion
sets: OSET1 (EDGE) or OSET2 (CSHD). OSET1 is derived from a set of 100 images
labeled in the JOV paper DiMattina, Fox, & Lewicki, (2012). There are two directories
with the training data: CV_SHAD_EDGE_BYIM, CV_SHAD_CSHD_BYIM. The files here are broken
up into 5 non-overlapping folds. When you train the models, you hold one back as a
test set and train on the remaining 4.

REQUIRES
--------
This was built on Tensorflow v2.6 and Keras v2.6 using PyCharm and Anaconda
on Windows 10 PCs. Although I have a GPU, training was done on the CPU in order
to obtain reproducible results (see online for details).

TRAINING
--------
To train the FRF model, type:

python frf_model_train2.py <occset> <nhidden> <testset> <l2> <outfile>

<occset>   - occlusion set. Values: 'EDGE' or 'CSHD'
<nhidden>  - number of hidden units
<testset>  - folder of images to hold back for testing. Values: 1-5
<l2>       - regularization parameter = 10^-l2. For instance if l2 = 2,
             the parameter is 10^-2 = 0.01
<outfile>  - filename for the KERAS model

To train the GFB model, you use the command:

python gabor_model_train2.py <occset> <testset> <l2> <outfile>

Both programs assume the training data exists in directories
../CV_SHAD_EDGE_BYIM
../CY_SHAD_CSHD_BYIM

These may be found in S1_Data.zip

CONTACT
-------
For additional questions, please e-mail: cdimattina@fgcu.edu
