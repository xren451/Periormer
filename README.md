# CDE-Periormer: CDE-Enhanced Periodic Transformer for irregularly sampled time series

This repository contains the code and datasets for the paper "CDE-Periormer: CDE-Enhanced Periodic Transformer for irregularly sampled time series". In this paper, we propose a new method for time series prediction considering different periodic patterns under different sampling rate of raw series.

We design CDE-Periormer structure to handle irregular sampling rate and periodic patterns. The overall architecture can be found on the following figure.

![1 Architecture](https://github.com/xren451/CDE-Periormer/blob/main/img/arch.png)

Also, the periodic attention is shown in the following figure.
![2 periatt](https://github.com/xren451/CDE-Periormer/blob/main/img/periodic%20att.png)

The main results can be found on the following table.
![3 periatt](https://github.com/xren451/CDE-Periormer/blob/main/img/Multivariate%20and%20univariate%20time%20series%20prediction%20on%20eight%20datasets.png)

# Dataset:
For WTH, ETTh1, ETTm1 and ETTm2, please refer to [here](https://github.com/zhouhaoyi/Informer2020) for detailed illustration.
For NDBC, please refer to [here](https://github.com/xren451/CDE-Periormer/blob/main/data/Environment/Methods%20to%20obtain%20NDBC%20datasets.txt) and [here] (https://github.com/xren451/CDE-Periormer/blob/main/data/Environment/ndbc_web_data_guide%20(1).pdf) for detailed data retrieving method. Please use [here](https://github.com/xren451/CDE-Periormer/blob/main/data/Environment/read_NDBC.ipynb) for a easy way to obtain the NDBC dataset.
For stock1 and stock2, please refer to [here](https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo) to obtain the dataset.

# Running the experiment:

To run the experiment, please use "example.sh" to run prediction experiment for WTH dataset.
