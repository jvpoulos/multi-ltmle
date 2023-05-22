# multi-ltmle

Longitudinal TMLE (LTMLE) with multi-valued treatments. 

N.b.: We cannot provide the actual Centers for Medicare & Medicaid Services (CMS) data used in the application because they are protected. The simulated data provided in this repo are for illustrative purposes.

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

* Required **R** packages located in ***package_list.R*** 

Contents
------


* ***package_list.R*** install required **R** packages.
	+ *doMPI*: logical flag. When TRUE, install packages needed for MPI parallel processing. Defaults to FALSE.

* ***src/misc_fns***: includes misc. functions, including a function to bound predicted probabilities; functions generate different distributions; and a forest plot function. 

* ***src/SL_fns***: pres-specified super learner functions for *ltmle* software. 

* ***src/tmle_calculation_long.R***: function for generating counterfactual means under each treatment rule in longitudial data. Inputs initial Y estimates, bounded cumulative treatment/censoring probabilities, observed treatment, and observed outcomes. Outputs treatment-rule specific means.

* ***src/tmleContrastLong.R***: function for calculating contrasts across all treatment rules in longitudinal data. Inputs treatment-rule-specific means, the contrast matrix, and logical flags. Outputs ATE and variance estimates. 

* ***simulation.R***: longitudinal setting (T>1) simulation, comparing the performance of manual multinomial TMLE with existing implementations using multiple binary treatments, with multiple levels of treatment. Simulates data over multiple runs and compares implementations in terms of bias, coverage, and CI width. The script consists of the following relevant parameters:

	+ *estimator*: Select which estimator to use: 'tmle' for multinomial and multiple binary TMLE using a standard super learner ensemble, 'tmle-lstm' for multinomial and multiple binary TMLE using an ensemble of LSTMs, 'lmtp' for TMLE with the *lmtp* package, 'ltmle' for TMLE with the *ltmle* package, 'iptw' for inverse probability of treatment weighting, 'gcomp' for g-computation, and 'sdr' for sequentially doubly-robust regression with the *lmtp* package. 

	+ *treatment.rule*: Treatment rule; can be "static", "dynamic", "stochastic", or "all" (if *estimator*='tmle')

	+ *gbound* and *ybound* numerical vectors defining bounds to be used for the propensity score and initial Y predictions, resp. Default is c(0.01,0.99) and c(0.0001, 0.9999), resp.  

	+ *J*: number of treatments; must be J=6.

	+ *n*: sample size. Defaults to 20000.

	+ *t.end*: number of time periods, must be at least 4 and no more than 36. Defaults to 36 (must be 36 if estimator='tmle').  

	+ *R*: number of simulation runs. Default is 1000. 

	+ *target.gwt*: logical flag. When TRUE, moves propensity weights from denominator of clever covariate to regression weight when fitting updated model for Y; used only for 'tmle' estimator. Default is TRUE. 

	+ *use.SL*: logical flag. When TRUE, use Super Learner for treatment and outcome model estimation; if FALSE, use GLM. 

	+ *n.folds*: number of cross-validation folds for Super Learner. Defaults to 10 and is ignored if *use.SL* is FALSE. 

* ***long_sim_plots.R*** combine output from ***simulation.R*** and plot.

* ***tmle_itt_analysis.R*** code for ITT analysis on actual CMS data in a static setting (T=1) and J=6 levels of treatment.
	+ ***tmle_itt_analysis_eda.R*** code for producing descriptive plots and tables for ITT analysis on actual CMS data in the static setting (T=1).

* ***ltmle_analysis.R*** code for analysis on actual CMS data in the longitudinal setting (T>1) and J=6 levels of treatment.

Instructions
------

1. Run in bash script: `Rscript simulation.R [arg1] [arg2] [arg3] [arg4]`; where `[arg1]` specifies the estimator ['tmle', 'lmtp','ltmle'], `[arg2]` is a number specifying the treatment rule [1-3, except if 'tmle', 1 should be used], and `[arg3]`  is a logical flag if super learner estimation is to be used ["TRUE" or "FALSE"], and `[arg4]` is a logical flag for using MPI parallel programming; e.g., 

`Rscript simulation.R 'tmle' 1 'TRUE' 'FALSE'`

2. Run in bash script: `Rscript ltmle_analysis.R [arg1] [arg2] [arg3] [arg4]`; where `[arg1]` specifies the estimator ['tmle', 'lmtp'], `[arg2]` is a character specifying the treatment rule ['static',dynamic','stochastic',or 'all' for estimator='tmle'], `[arg3]` is a string that specified the folder of previously saved weights or 'none', and `[arg4]` is a logical flag if super learner estimation is to be used; e.g, 

`Rscript ltmle_analysis.R 'tmle' 'all' '20230329/' 'TRUE'`  