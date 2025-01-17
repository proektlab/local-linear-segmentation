# local-linear-segmentation
This repository contains the main scripts for local linear segmentation and subsequent analysis of the resulting model space presented in:

Costa AC, Ahamed T, Stephens GJ (2019) "Adaptive, locally-linear models of complex dynamics" PNAS https://doi.org/10.1073/pnas.1813476116

Any comments or questions, contact antoniocbscosta(at)gmail(dot)com. Also, suggestions to speed up the code are more than welcome!

As good practice, we advise users to make sure their data is appropriate before applying this method. Post-processing to reduce the amount of spurious or noisy data in your time series will ensure that the results are more easily interpretable. In addition, choosing the right measurements can be of crucial importance. If your time series is multi-dimensional, try to make sure that the different dimensions are not linearly dependent. Our method ensures that linear reression on the shortest window is well-conditioned, so getting rid of collinearity as much as possible is an advantage, since it allows to probe shorter time scales.


-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------


To run the segmentation algorithm, the following steps must be followed:

1. - Install the following packages:

- scipy, numpy, matplotlib
- cython, scikit-learn, h5py

(tested with scipy-1.1.0, numpy-1.15.2, matplotlib-3.0.0, cython-0.28.5, scikit-learn-0.20.0 and h5py-2.8.0)


2. - run the 'setup.py' file which compiles the 'LLSA_calculations.pyx' cython script ('setup.py' is in the 'segmentation_code' folder)

python setup.py build_ext --inplace


In order to the able to run the .ipynb tutorials, you'll also need jupyter and seaborn.


(a) - 'SegmentingHO.ipynb' and (b) - 'SegmentingWormBehavior.ipynb' are two complementary tutorials on how to apply the adaptive locally-linear segmentation. In (a), a toy time series is segmented and hierarchical clustering is applied to obtain the original model parameters. In (b), a sample C. elegans "eigenworm" time series is analysed, in which the worm is subject to a heat shock to the head that triggers an escape response, [1]. The sampled time series start at the initiation of the escape response, which is broadly composed of a reversal, a turn, and forward movement in a different direction, away from the stimulus. 

For applications in high-dimensional systems, the conservative minimum window size obtained using the condition number threshold on the entire time series might be too long, depending on the inherent correlation time of the dynamics and the sampling rate. In this situation, regularization can be added on short time scales. An example of the implementation of such a regularization is in 'SegmentingVisAl_regularization.ipynb'.



-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------


'TestSegmentationScript.py' is a full python script that segments the toy time series. It takes various parameters as arguments
	- n: size of the null distribution
	- min_w: minimum window size
	- s: step fraction in the definition of the minimum window size
	- per: percentile of the null likelihood distribution that defines the threshold significance level (2x(100-per))

e.g.,

python TestSegmentationScript.py -n 1000 -min_w 6 -s 0.1 -per 97.5

'TestSegmentationScript.py' applies the segmentation algorithm to the time series from 'SampleTseries.h5'. The details of the simulation can be found in the 'MetaData' folder of 'SampleTseries.h5' and for more intuition follow the 'SegmentingHO.ipynb' tutorial.

'LLSA.py' contains the main skeleton of the segmentation algorithm.

'LLSA_calculations.pyx' is a cython script with the main functions used in the segmentation (such as "get_theta" which fits a linear model, or "R_null" which draws the null likelihood distribution).

The resulting model space can be analysed by likelihood hierarchical clustering using 'Distance_calculations.py', which computes the likelihood distance matrix, performs hierarchical clustering, and returns the corresponding models.


-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------


The analysis pipeline can be significantly sped-up by introducing parallel processing in a few places. Examples include:

1) Likelihood ratio distribution: we need to do the exact same calculation N times

2) Parallelize over different time series, or pre-split the time series into chunks and parallelize over them

3) In the calculation of the distance matrix we can both parallelize one of the loops and also only compute the upper triangular or lower triangular matrix since, by construction, the matrix will be symmetric

For a discussion of such implementations see 'SegmentationHO.ipynb'.


-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------


The data contained in '\sample_data' consists of three samples in different systems:

1) 'AllenData_sample.h5' is a snippet of Calcium imaging of hundreds of neurons in the mouse VisAl cortex. Data was obtained from the Allen Institute [2] (experiment 511854338). 

2) 'Sample_tseries.h5' is a simulation of a set of harmonic oscillators that reverse direction.

3) 'worm_tseries.h5' is a snippet of a laser induced escape response of C. elegans. The full dataset is published in the Dryad Digital Repository [3].

[1] - Broekmans O, Rodgers J, Ruy S, Stephens GJ (2016) "Resolving coiled shapes reveals new reorientation behaviors in C. elegans" eLife 2016;5:e17227; https://doi.org/10.7554/elife.17227

[2] - Allen Institute for Brain Science (2016) Allen Brain Observatory. Available at http://observatory.brain-map.org/visualcoding/

[3] - Broekmans OD, Rodgers JB, Ryu WS, Stephens GJ (2016) Data from: Resolving coiled shapes reveals new reorientation behaviors in C. elegans. Dryad Digital Repository. https://doi.org/10.5061/dryad.t0m6p
