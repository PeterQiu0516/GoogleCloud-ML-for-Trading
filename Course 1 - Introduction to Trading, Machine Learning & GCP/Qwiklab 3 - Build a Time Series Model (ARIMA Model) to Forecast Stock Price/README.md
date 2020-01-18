# Qwiklab 3 - Build a Time Series Model (ARIMA Model) to Forecast Stock Price

## Lab Manual
>[Qwiklab 3 - Lab Manual (Preview)](https://github.com/PeterQiu0516/GoogleCloud-ML-for-Trading/blob/master/Course%201%20-%20Introduction%20to%20Trading%2C%20Machine%20Learning%20%26%20GCP/Qwiklab%203%20-%20Build%20a%20Time%20Series%20Model%20(ARIMA%20Model)%20to%20Forecast%20Stock%20Price/Qwiklab%203%20-%20Lab%20Manual.pdf)

>[Qwiklab 3 - Lab Manual (Download)](https://github.com/PeterQiu0516/GoogleCloud-ML-for-Trading/raw/master/Course%201%20-%20Introduction%20to%20Trading%2C%20Machine%20Learning%20%26%20GCP/Qwiklab%203%20-%20Build%20a%20Time%20Series%20Model%20(ARIMA%20Model)%20to%20Forecast%20Stock%20Price/Qwiklab%203%20-%20Lab%20Manual.pdf)

## About
This lab is based on Qwiklabs Supported by [Google Cloud Platform(GCP)](https://cloud.google.com/). 

Main Webpage for Qwiklabs: https://www.qwiklabs.com/

In this lab, I build a Time Series Forecasting Model **(ARIMA Model)** by Python using the `statsmodels` library to predict AAPL stock prices.

## Running
Main file is the Jupyter Notebook file named `arima_model.ipynb`.

In order to execute it, first run:
```
pip install jupyter
```
and also run:
```
pip install statsmodels
```

## Explore More
### Time Series
A time series is a series of data points indexed (or listed or graphed) in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data. Examples of time series are heights of ocean tides, counts of sunspots, and the daily closing value of the Dow Jones Industrial Average.

Time series analysis comprises methods for analyzing time series data in order to **extract meaningful statistics** and other characteristics of the data. 

Time series forecasting is the use of a model to **predict future values based on previously observed values**.

### ARIMA model

An **autoregressive integrated moving average (ARIMA) model** is a generalization of an autoregressive moving average (ARMA) model, which is fitted to time series data either to *better understand the data* or to *predict future points* in the series (forecasting). 

ARIMA models are applied in some cases where data show evidence of non-stationarity, where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity.

The **AR (Auto-Regressive)** part of ARIMA indicates that the evolving variable of interest is *regressed on its own lagged (i.e., prior) values*. 

The **MA (Moving Average)** part indicates that the regression error is actually a linear combination of error terms whose values occurred contemporaneously and at various times in the past. 

The **I (Integration)** indicates that the data values have been replaced with the difference between their values and the previous values (and this differencing process may have been performed more than once). The purpose of each of these features is to make the model fit the data as well as possible.
