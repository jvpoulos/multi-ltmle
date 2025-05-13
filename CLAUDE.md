# Commands and Guidelines for multi-ltmle

## Build/Test Commands
- Install packages: `Rscript package_list.R`
- Run simulation with fixes:
  - Preferred method: `./run_simulation.sh [estimator] [cores] [use_sl] [do_mpi]`
  - Standard tmle estimator: `./run_tmle.sh`
  - TMLE-LSTM estimator: `./run_tmle_lstm.sh`
  - Direct R call: `Rscript run_simulation.R 'tmle-lstm' 1 'TRUE' 'FALSE'`
- Plot results: `Rscript long_sim_plots.R 'outputs/YYYYMMDD'`
- Test LSTM model: `Rscript test_predictions_in_r.R`
- Python environment: `source myenv/bin/activate` before running R

## Error Fixes
- General fixes (for all estimators): `source("fix_simulation.R")`
- TMLE-specific fixes: `source("makeshift_tmle.R")`
- See HOW_TO_RUN.md for details on specific errors and their fixes

## Code Style Guidelines
- **R naming:** snake_case for functions/variables, CamelCase for classes
- **Python naming:** snake_case for functions/variables, PascalCase for classes
- **Documentation:** Roxygen style for R functions, docstrings for Python
- **Error handling:** Use tryCatch in R, try/except in Python with informative messages
- **Imports:** Group imports by type (base, then third-party, then local)
- **File paths:** Always use absolute paths with file.path() or os.path.join()
- **R/Python integration:** Always use proper error handling with reticulate