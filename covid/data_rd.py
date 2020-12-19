"""
This module contains all US-specific data loading and data cleaning routines.
"""
import requests
import pandas as pd
import numpy as np

idx = pd.IndexSlice


def get_raw_covidtracking_data():
    """ Gets the current daily CSV from COVIDTracking """
    url = "https://raw.githubusercontent.com/gcaff/COVID19-RD/master/data/covid_data_rd.csv"
    data = pd.read_csv(url, encoding='latin-1')
    return data


def process_covidtracking_data(data: pd.DataFrame, run_date: pd.Timestamp):
    """ Processes raw COVIDTracking data to be in a form for the GenerativeModel.
        In many cases, we need to correct data errors or obvious outliers."""
    data = data.rename(columns={"provincia": "region"})
    data = data.rename(columns={"casos_acum": "positive"})
    data = data.rename(columns={"fecha": "date"})
    data = data.rename(columns={"procesadas": "total"})
    data["date"] = pd.to_datetime(data["date"], format="%d/%m/%Y")
    data = data.set_index(["region", "date"]).sort_index()
    data = data[["positive", "total"]]

    # Now work with daily counts
    data = data.diff().dropna().clip(0, None).sort_index()
    
    zero_filter = (data.positive >= data.total)
    data.loc[zero_filter, :] = 0
    data.loc[idx["La Romana", pd.Timestamp("2020-12-02")], :] = 0


    # At the real time of `run_date`, the data for `run_date` is not yet available!
    # Cutting it away is important for backtesting!
    return data.loc[idx[:, :(run_date - pd.DateOffset(1))], ["positive", "total"]]


def get_and_process_covidtracking_data(run_date: pd.Timestamp):
    """ Helper function for getting and processing COVIDTracking data at once """
    data = get_raw_covidtracking_data()
    data = process_covidtracking_data(data, run_date)
    return data
