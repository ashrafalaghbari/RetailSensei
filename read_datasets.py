import pandas as pd
import pandas as pd
from fredapi import Fred
import streamlit as st
import requests
import os
import csv


# @st.cache
class Datasets:
    def __init__(self):
       # train data from 1992-01-01 to 2021-04-01 
        self.train = pd.read_csv('datasets/train.csv', index_col=0, parse_dates=True)
        self.train = self.train.squeeze()
        self.train.index.freq = 'MS'

        # test data from 2021-05-01 to 2023-04-01
        self.test = pd.read_csv('artifacts/test.csv', index_col=0, parse_dates=True)
        self.test = self.test.squeeze()
        self.test.index.freq = 'MS'

        # Read the entire data (train and test) from 1992-01-01 to 2023-04-01
        self.entire_data = pd.read_csv('datasets/entire_data.csv', index_col=0, parse_dates=True)
        self.entire_data = self.entire_data.squeeze()
        self.entire_data.index.freq = 'MS'

        # Read trend data that was decomposed with CMA (Centered Moving Average)
        self.cma = pd.read_csv('datasets/cma.csv', index_col=0, parse_dates=True)
        self.cma = self.cma.squeeze()

        # read the seasonal indices series that includes the seasonal indices for each month from 1992-01-01 to 2023-04-01
        self.seasonal_indices_series = pd.read_csv('datasets/seasonal_indices_series.csv', index_col=0, parse_dates=True)
        self.seasonal_indices_series = self.seasonal_indices_series.squeeze()
        self.seasonal_indices_series.index.freq = 'MS'

        # read the residuals (what wasn't captured by trend and seasonality)
        self.residuals = pd.read_csv('datasets/residuals.csv', index_col=0, parse_dates=True)
        self.residuals = self.residuals.squeeze()
        self.residuals.index.freq = 'MS'

        # read the seasonal indices dataframe that includes seasoanl value, change in seasonal value, and the percentage change in seasonal value
        # The dataframe is used to plot the seasonal indices
        self.seasonal_indices_df = pd.read_csv('datasets/seasonal_indices.csv', index_col=0, parse_dates=True)
        self.seasonal_indices_df = self.seasonal_indices_df.squeeze()
        self.seasonal_indices_df.index.freq = 'MS'

        # forecast generated during model development from 2021-05-01 to 2023-04-01
        self.hw_forecast_dev = pd.read_csv('datasets/hw_forecast_dev.csv', index_col=0, parse_dates=True)
        self.hw_forecast_dev = self.hw_forecast_dev.squeeze()
        self.hw_forecast_dev.index.freq = 'MS'
        self.hw_forecast_dev_mean = self.hw_forecast_dev.iloc[:, 0]
        self.hw_forecast_dev_lower = self.hw_forecast_dev.iloc[:, 2]
        self.hw_forecast_dev_upper = self.hw_forecast_dev.iloc[:, 3]

        # read the saved forecast for the next 24 months after 2023-04-01 from (2023-05-01 to 2025-04-01)
        self.hw_forecast = pd.read_csv('datasets/hw_forecast.csv', index_col=0, parse_dates=True)
        self.hw_forecast = self.hw_forecast.squeeze()
        self.hw_forecast.index.freq = 'MS'
        self.hw_forecast_mean = self.hw_forecast.iloc[:, 0]
        self.hw_forecast_lower = self.hw_forecast.iloc[:, 2]
        self.hw_forecast_upper = self.hw_forecast.iloc[:, 3]

    # function to return the relevant time series
    def get_datasets(self):
        return (self.train, self.test, self.entire_data, self.cma, self.seasonal_indices_series, self.residuals,
            self.seasonal_indices_df, self.hw_forecast_dev_mean,
            self.hw_forecast_dev_lower, self.hw_forecast_dev_upper, self.hw_forecast_mean,
            self.hw_forecast_lower, self.hw_forecast_upper, self.hw_forecast)
