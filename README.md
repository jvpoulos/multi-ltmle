# multi-ltmle

Longitudinal TMLE (LTMLE) with multi-valued treatments. 

N.b.: We cannot provide the actual Centers for Medicare & Medicaid Services (CMS) data used in the application because they are protected. The simulated data provided in this repo are for illustrative purposes.

Please cite the following papers if you use this repo:

```
@article{https://doi.org/10.1002/sim.10003,
  title={Targeted learning in observational studies with multi-valued treatments: An evaluation of antipsychotic drug treatment safety.},
  author={Poulos, Jason and Horvitz-Lennon, Marcela and Zelevinsky, Katya and Cristea-Platon, Tudor and Huijskens, Thomas and Tyagi, Pooja and Yan, Jiaju and Diaz, Jordi and Normand, Sharon-Lise},
  journal={Statistics in Medicine},
  year={2024},
   publisher={Wiley Online Library}
}
```

```
@article{https://doi.org/10.1017/S0033291723001502,
  title={Antipsychotics and the risk of diabetes and death among adults with serious mental illnesses},
  author={Poulos, Jason and Normand, Sharon-Lise T and Zelevinsky, Katya and Newcomer, John W and Agniel, Denis and Abing, Haley K and Horvitz-Lennon, Marcela},
  journal={Psychological Medicine},
  volume={53},
  number={16},
  pages={7677--7684},
  year={2023},
  publisher={Cambridge University Press}
}
```

Prerequsites
------

* **R** (tested on 4.0.1 using a 6.2.0 GCC compile)
+ Required **R** packages located in ***package_list.R*** 

* For use of 'tmle-lstm' as an estimator: **R** (tested on 4.3.1), **python3** (tested on 3.6.8), and **TensorFlow** (tested on 2.12.0) using a GCC 9.2.0 compiler and cuda 11.2 for GPU computation
+ instructions for installing Tensorflow on Linux (documentation [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) and [here](https://www.tensorflow.org/install/pip#linux))
```
# create virtual environment within directory
cd multi-ltmle
python3 -m venv env
source env/bin/activate

# install Tensorflow and TensorRT
pip install --upgrade pip
pip install tensorflow-gpu
# pip install tensorflow-cpu # for CPU-only build
pip install tensorrt

python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))" # verify the CPU setup
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" # verify the GPU setup

# start R in virtual environment where python3 and Tensorflow are installed
R
```
+ The following Python packages are required: numpy (tested on 1.19.5) and pandas (tested on 1.1.5)
```
pip install numpy
pip install pandas
```
+ Additional **R** packages are required by setting the *keras* flag in ***package_list.R*** to TRUE

Contents
------

* ***package_list.R*** install required **R** packages.
	+ *doMPI*: logical flag. When TRUE, install packages needed for MPI parallel processing. Defaults to FALSE.
	+ *keras*: logical flag. When TRUE, install packages needed for Keras/TensorFlow. Defaults to FALSE.

* ***src/misc_fns***: includes misc. functions, including a function to bound predicted probabilities; functions generate different distributions; and a forest plot function. 

* ***src/simcausal_fns***: defines distribution functions for *simcausal* software.

* ***src/simcausal_dgp***: defines data generating process for *simcausal* software.

* ***src/tmle_fns***: define treatment rule functions for our implementation. 

* ***src/tmle_calculation_long.R***: function for generating counterfactual means under each treatment rule in longitudinal data. Inputs initial Y estimates, bounded cumulative treatment/censoring probabilities, observed treatment, and observed outcomes. Outputs treatment-rule specific means.

* ***src/tmleContrastLong.R***: function for calculating contrasts across all treatment rules in longitudinal data. Inputs treatment-rule-specific means, the contrast matrix, and logical flags. Outputs ATE and variance estimates. 

* ***src/lstm.R***: function for calling Python code for estimating with LSTMs within R; used when estimator ='tmle-lstm'. 

* ***src/train_lstm.py***: Python code for training LSTMs and predicting on the same data; used when estimator ='tmle-lstm'.

* ***src/test_lstm.py***: Python code for inference using a trained LSTMs model on new data; used when estimator ='tmle-lstm'.

* ***simulation.R***: longitudinal setting simulation, comparing the performance of manual multinomial TMLE with existing implementations using multiple binary treatments, with multiple levels of treatment. Simulates data over multiple runs and compares implementations in terms of bias, coverage, and CI width. The script consists of the following relevant parameters:

	+ *estimator*: Select which estimator to use: 'tmle' for our TMLE implementation (multinomial and multiple binary),  using a standard super learner ensemble (also returns estimates from an inverse probability of treatment weighting, IPTW, estimator and g-computation estimator); 'tmle-lstm' for multinomial and multiple binary TMLE using an ensemble of LSTMs (also returns IPTW and g-computation estimates).

	+ *treatment.rule*: Treatment rule; can be "static", "dynamic", "stochastic", or "all" (if *estimator*='tmle')

	+ *gbound* and *ybound* numerical vectors defining bounds to be used for the propensity score and initial Y predictions, resp. Default is c(0.05,1)  and c(0.0001,0.9999), resp.

	+ *J*: number of treatments; must be J=6.

	+ *n*: sample size. Defaults to 12500.

	+ *t.end*: number of time periods, must be at least 4 and no more than 36. Defaults to 36 (must be 36 if estimator='tmle').  

	+ *R*: number of simulation runs. Default is 325. 

	+ *target.gwt*: logical flag. When TRUE, moves propensity weights from denominator of clever covariate to regression weight when fitting updated model for Y; used only for 'tmle' estimator. Default is TRUE. 

	+ *use.SL*: logical flag. When TRUE, use Super Learner for treatment and outcome model estimation; if FALSE, use GLM (**use.SL=FALSE not functional**). 

	+ *scale.continuous*: logical flag. When TRUE, use scale continuous variables before training. Default is FALSE. 

	+ *n.folds*: number of cross-validation folds for Super Learner. Defaults to 5 (must be at at least 3). 

* ***long_sim_plots.R*** combine output from ***simulation.R*** and plot.

* ***ltmle_analysis.R*** code for analysis on actual CMS data in the longitudinal setting (T>1) and J=6 levels of treatment.
	+ ***add_tv_simulated.R*** code for quickly generating time-varying variables in the simulated dataset (for illustrative purposes)
	* ***ltmle_analysis_eda.R*** code for producing descriptive plots and tables for the analysis. 

Instructions
------

1. Install require **R** packages: `Rscript package_list.R` and follow Python installation instructions in the Prerequsites section.

2. For simulations, run: `Rscript simulation.R [arg1] [arg2] [arg3] [arg4]`; where `[arg1]` specifies the estimator ["tmle", "tmle-lstm"], `[arg2]` is a number specifying the treatment rule [1 for all treatment rules should be used], and `[arg3]`  is a logical flag if super learner estimation is to be used ["TRUE" or "FALSE"], and `[arg4]` is a logical flag for using MPI parallel programming; e.g., 

	`Rscript simulation.R 'tmle' 1 'TRUE' 'FALSE'`

3. To plot simulation results, run: `Rscript long_sim_plots.R [arg1]`; where `[arg1]` specifies the output path of the simulation results. E.g., 
	
	`Rscript long_sim_plots.R 'outputs/20240215'`

4. Download in the local directory simulated data from [simdata_from_basevars.RData](https://github.com/jvpoulos/multi-tmle/blob/4286f7899ec0a9fc27474ff88871dbd6cae85dbd/simdata_from_basevars.RData) These simulated data are for illustrative purposes and are provided in the cross-sectional study repo [multi-tmle](https://github.com/jvpoulos/multi-tmle/). The file ***add_tv_simulated.R*** quickly generates time-varying covariates to demonstrate for the longituninal analysis.

5. For ITT analysis on simulated data, run: `Rscript ltmle_analysis.R [arg1] [arg2] [arg3] [arg4] [arg5]`; where `[arg1]` specifies the estimator ["tmle", "tmle-lstm"], `[arg2]` is a character specifying the treatment rule ['all' for estimating 'static',dynamic', and 'stochastic' rules at once], `[arg3]` is a string that specified the folder of previously saved weights (e.g., '20240215/') or 'none', `[arg4]` is a logical flag if super learner estimation is to be used, and , `[arg5]` is a logical flag if simulated data is to be used; e.g, 

`Rscript ltmle_analysis.R 'tmle' 'all' 'none' 'TRUE' 'TRUE'`

CPU computation is used if "tmle-lstm" is selected for the estimator.

Simulation example
------

Example output images are saved in `outputs/`. The following are from a single simulated longitudinal dataset for n=10k people. We estimate the counterfactual diabetes risk for those who continued to follow each of 3 regimes (static, dynamic, stochastic) in each of 36 time periods, based on the observed outcomes and covariates. 

![demo](./ex_outputs/treatment_adherence_10000.png)

![demo](./ex_outputs/survival_plot_observed_10000.png)

![demo](./ex_outputs/survival_plot_truth_10000.png)

![demo](./ex_outputs/survival_plot_tmle_estimates_10000.png)