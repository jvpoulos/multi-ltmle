# multi-ltmle

Longitudinal TMLE (LTMLE) with multi-valued treatments. 

N.b.: We cannot provide the actual Centers for Medicare & Medicaid Services (CMS) data used in the application because they are protected. The simulated data are for illustrative purposes.

Please cite the paper if you use this repo:

```
@misc{https://doi.org/10.48550/arxiv.2206.15367,
  doi = {10.48550/ARXIV.2206.15367},
  url = {https://arxiv.org/abs/2206.15367},
  author = {Poulos, Jason and Horvitz-Lennon, Marcela and Zelevinsky, Katya and Cristea-Platon, Tudor and Huijskens, Thomas and Tyagi, Pooja and Yan, Jiaju and Diaz, Jordi and Normand, Sharon-Lise},
  keywords = {Applications (stat.AP), Methodology (stat.ME), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Targeted learning in observational studies with multi-level treatments: An evaluation of antipsychotic drug treatment safety},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```


Prerequsites
------

* **R** (tested on 4.0.1)
	+ Required **R** packages located in ***package_list.R*** 

* **python3** (tested on 3.11.2) and **TensorFlow** (tested on 2.12.1) for use of 'tmle-lstm' as an estimator
	* see documentation: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/ and https://www.tensorflow.org/install/pip#linux
		+ start R in virtual environment where python3 and Tensorflow are installed
	+ *keras* flag in ***package_list.R*** must be set to TRUE

Contents
------

* ***package_list.R*** install required **R** packages.
	+ *doMPI*: logical flag. When TRUE, install packages needed for MPI parallel processing. Defaults to FALSE.
	+ *keras*: logical flag. When TRUE, install packages needed for Keras/TensorFlow. Defaults to FALSE.

* ***src/misc_fns***: includes misc. functions, including a function to bound predicted probabilities; functions generate different distributions; and a forest plot function. 

* ***src/simcausal_fns***: defines distribution functions for *simcausal* software.

* ***src/simcausal_dgp***: defines data generating process for *simcausal* software.

* ***src/SL_fns***: pre-specified super learner functions and library definitions for *ltmle* software.

* ***src/lmtp_fns***: define treatment rule functions for *lmtp* software. 

* ***src/ltmle_fns***: define treatment rule functions for *ltmle* software. 

* ***src/tmle_fns***: define treatment rule functions for our implementation. 

* ***src/tmle_calculation_long.R***: function for generating counterfactual means under each treatment rule in longitudinal data. Inputs initial Y estimates, bounded cumulative treatment/censoring probabilities, observed treatment, and observed outcomes. Outputs treatment-rule specific means.

* ***src/tmleContrastLong.R***: function for calculating contrasts across all treatment rules in longitudinal data. Inputs treatment-rule-specific means, the contrast matrix, and logical flags. Outputs ATE and variance estimates. 

* ***simulation.R***: longitudinal setting (T>1) simulation, comparing the performance of manual multinomial TMLE with existing implementations using multiple binary treatments, with multiple levels of treatment. Simulates data over multiple runs and compares implementations in terms of bias, coverage, and CI width. The script consists of the following relevant parameters:

	+ *estimator*: Select which estimator to use: 'tmle' for our TMLE implementation (multinomial and multiple binary),  using a standard super learner ensemble (also returns estimates from an inverse probability of treatment weighting, IPTW, estimator and g-computation estimator); 'tmle-lstm' for multinomial and multiple binary TMLE using an ensemble of LSTMs (also returns IPTW and g-computation estimates); 'lmtp-tmle' for TMLE with the *lmtp* package; 'lmtp-iptw' for IPTW with the *lmtp* package ; 'lmtp-gcomp' for g-computation with the *lmtp* package; 'lmtp-sdr' for sequentially doubly-robust regression (SDR) with the *lmtp* package; 'ltmle-tmle' for TMLE with the *ltmle* package (also returns IPTW estimates); and  'ltmle-gcomp' for g-computation with the *ltmle* package. **estimator='tmle' is currently the only functional option**.

	+ *treatment.rule*: Treatment rule; can be "static", "dynamic", "stochastic", or "all" (if *estimator*='tmle')

	+ *gbound* and *ybound* numerical vectors defining bounds to be used for the propensity score and initial Y predictions, resp. Default is c(0.025,1)  and c(0.0001,0.9999), resp.

	+ *J*: number of treatments; must be J=6.

	+ *n*: sample size. Defaults to 10000.

	+ *t.end*: number of time periods, must be at least 4 and no more than 36. Defaults to 36 (must be 36 if estimator='tmle').  

	+ *R*: number of simulation runs. Default is 5. 

	+ *target.gwt*: logical flag. When TRUE, moves propensity weights from denominator of clever covariate to regression weight when fitting updated model for Y; used only for 'tmle' estimator. Default is TRUE. 

	+ *use.SL*: logical flag. When TRUE, use Super Learner for treatment and outcome model estimation; if FALSE, use GLM (**use.SL=FALSE not functional**). 

	+ *n.folds*: number of cross-validation folds for Super Learner. Defaults to 5 (must be at at least 3). 

* ***long_sim_plots.R*** combine output from ***simulation.R*** and plot.

* ***ltmle_analysis.R*** code for analysis on actual CMS data in the longitudinal setting (T>1) and J=6 levels of treatment.
	+ ***add_tv_simulated.R*** code for quickly generating time-varying variables in the simulated dataset (for illustrative purposes)

Instructions
------

1. Install require **R** packages: `Rscript package_list.R`

2. For simulations, run: `Rscript simulation.R [arg1] [arg2] [arg3] [arg4]`; where `[arg1]` specifies the estimator ["lmtp-tmle","lmtp-iptw","lmtp-gcomp","lmtp-sdr","ltmle-tmle","ltmle-gcomp","tmle", "tmle-lstm"], `[arg2]` is a number specifying the treatment rule [1-3, except if 'tmle', 1 should be used], and `[arg3]`  is a logical flag if super learner estimation is to be used ["TRUE" or "FALSE"], and `[arg4]` is a logical flag for using MPI parallel programming; e.g., 

`Rscript simulation.R 'tmle' 1 'TRUE' 'FALSE'`

3. To plot simulation results, run: `Rscript long_sim_plots.R [arg1]`; where `[arg1]` specifies the output path of the simulation results. E.g., 
	
	`Rscript long_sim_plots.R 'outputs/20230330'`

4. Download in the local directory simulated data from [simdata_from_basevars.RData](https://github.com/jvpoulos/multi-tmle/blob/4286f7899ec0a9fc27474ff88871dbd6cae85dbd/simdata_from_basevars.RData) These simulated data are for illustrative purposes and are provided in the cross-sectional study repo [multi-tmle](https://github.com/jvpoulos/multi-tmle/). The file ***add_tv_simulated.R*** quickly generates time-varying covariates to demonstrate for the longituninal analysis.

5. For ITT analysis on simulated data, run: `Rscript ltmle_analysis.R [arg1] [arg2] [arg3] [arg4] [arg5]`; where `[arg1]` specifies the estimator ["lmtp-tmle","lmtp-iptw","lmtp-gcomp","lmtp-sdr","ltmle-tmle","ltmle-gcomp","tmle", "tmle-lstm"], `[arg2]` is a character specifying the treatment rule ['static',dynamic','stochastic',or 'all' for estimator='tmle'], `[arg3]` is a string that specified the folder of previously saved weights (e.g., '20230329/') or 'none', `[arg4]` is a logical flag if super learner estimation is to be used, and , `[arg5]` is a logical flag if simulated data is to be used; e.g, 

`Rscript ltmle_analysis.R 'tmle' 'all' 'none' 'TRUE' 'TRUE'`  