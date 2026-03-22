# FiberSim_ML

This repository trains simple PyTorch regressors to predict FiberSim time-series outputs from simulation metadata and time.

## Training modes

- `train_from_readdata.py`: trains a model using 1 metadata variable plus time as input. It reads data from `data2/` by default.
- `train_from_readdata_n_var3.py`: trains a model using 3 metadata variables plus time as input. It reads data from `data_n_var3/` by default.

Both scripts:

- read tab-delimited target files and matching `_metadata.txt` files
- build point-wise training samples from full time-series data
- scale inputs and outputs with `MinMaxScaler`
- train a small fully connected neural network in PyTorch
- save a checkpoint, loss curve, and evaluation plot for held-out validation samples

## Main files

- `readdata.py`: helper functions for reading tab-delimited text files
- `timeseries_model_cpu.pt`: saved checkpoint for the 1-variable model
- `timeseries_model_n_var3_cpu.pt`: saved checkpoint for the 3-variable model
- `loss_curve.png` and `loss_curve_n_var3.png`: training and validation loss plots
- `evaluation_timeseries_validation_cases.png` and `evaluation_timeseries_validation_cases_n_var3.png`: predicted vs. actual validation plots

## Run

Install dependencies with your preferred environment manager, then run:

```powershell
python train_from_readdata.py
python train_from_readdata_n_var3.py
```

Each script trains and then immediately evaluates the saved model using its default dataset and settings.
