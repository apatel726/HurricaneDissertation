"""
Created on Tue Sep 15 01:04:55 2020
@author: Hammad, Akash, Jonathan
Scientific units used are as follows,
Coordinates (Lat, Lon) : Decimal Degrees (DD)
Timestamp : Python Datetime
Barometric pressure : mb
Wind Intensity: Knots
"""

import os
import fire
import pandas as pd
import pickle
import json
import csv
import hurricane_ai.plotting_utils
from typing import List, Dict
from hurricane_ai.ml.bd_lstm_td import BidrectionalLstmHurricaneModel
from ingest import *

def prep_hurricane_data(observations: List, lag: int) -> pd.DataFrame:
    """
    Converts raw observations to data frame and computes derived features.
    :param observations: Raw hurricane kinematic and barometric measurements.
    :param lag: Number of observation intervals to lag derived features.
    :return: Data frame of raw and derived hurricane measurements.
    """

    # Construct data frame from observations and sort by time
    df = pd.DataFrame(observations).sort_values(by="time")

    # TODO: This assumes everything is UTC - not sure if this is actually the case
    df["time"] = pd.to_datetime(df["time"], utc=True)

    df = df.assign(

        # Maximum wind speed up to time of observation
        max_wind=df["wind"].cummax(),

        # Change in wind speed since beginning of five day interval
        delta_wind=(df["wind"].cummax() - df["wind"].shift(lag).cummax()) / (
                (df["time"] - df["time"].shift(lag)).dt.seconds / 43200),

        # Minimum pressure up to time of observation
        min_pressure=df["pressure"].cummin(),

        # Average change in latitudinal position per hour
        zonal_speed=(df["lat"] - df["lat"].shift(lag)) / ((df["time"] - df["time"].shift(lag)).dt.seconds / 3600),

        # Average change in longitudinal position per hour
        meridonal_speed=(df["lon"] - df["lon"].shift(lag)) / (
                (df["time"] - df["time"].shift(lag)).dt.seconds / 3600),

        # Year/month/day/hour
        year=df["time"].dt.year,
        month=df["time"].dt.month,
        day=df["time"].dt.day,
        hour=df["time"].dt.hour
    )

    # Remove rows where we didn't have enough historical data to compute derived features
    df = df.dropna()
    
    return df

def run_live_inference(base_directory: str, model_file: str, scaler_file: str) -> None:
    """
    Pulls live storm data and runs single pass inference for every storm.
    :param base_directory: Path to directory containing serialized artifacts (e.g. models, scalers).
    :param model_file: Filename of the model file.
    :param scaler_file: Filename of the scaler file.
    """
    # load current model configuration
    with open(os.path.join(base_directory, 'hyperparameters.json')) as f:
        root = json.load(f)
        typesearch = root["config"]["name"]
        if root['universal'] :
            model_type = "universal"
        else:
            if typesearch == "sequential":
                model_type = "wind"
            elif typesearch == 'sequential_1':
                model_type = "lat"
            else:
                model_type = "long"
                    
        print(model_type)
        
    # 5 day lag
    lag = 5

    # Initialize model
    model = BidrectionalLstmHurricaneModel((None, None), model_type , os.path.join(base_directory, scaler_file),
                                           model_path=os.path.join(base_directory, model_file))
    # Grab live storm data
    live_storms = nhc()

    for storm in live_storms:
        print(f"Running inference for {storm['metadata']['name']}")

        # Build data frame with raw observations and derived features
        df = prep_hurricane_data(storm["entries"], lag)
        
        if (len(storm["entries"])) <= 20 : # 1 entry = 6 hours, 20 entries is 120 hours (5 days)
            print(f'{storm["metadata"]["name"]} does not have enough data (minimum 5 days)')
            continue
        
        # Run inference on the given observations
        result = model.predict(df, lag)
        print('-------------------------------------')
        
        # Converts the scaled values  from the model and scaler chosen to real values       
        with open(os.path.join(base_directory, scaler_file), 'rb') as f:
          scaler = pickle.load(f)
        
        # Run inference based on type of model
        if model_type == "universal" :
            for day in range(3) : # 3 days
                wind_index = 0
                lat_index = 1
                long_index = 2
                
                # wind prints the wind for the first 3 days with an input shape of 11 features
                wind_result = []
                wind_result.append(hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 2, result[day][wind_index])) 
                print(f'{day + 1} day: universal result wind test:{scaler.inverse_transform(wind_result)[0][2]}')
                
                # lat prints the wind for the first 3 days with an input shape of 11 features
                lat_result = []
                lat_result.append(hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 0, result[day][lat_index]))
                print(f'{day + 1} day: universal result lat test:{scaler.inverse_transform(lat_result)[0][0]}')
                
                # long prints the wind for the first 3 days with an input shape of 11 features
                long_result = []
                long_result.append(hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 1, result[day][long_index]))
                print(f'{day + 1} day: universal result long test:{scaler.inverse_transform(long_result)[0][1]}')
            
        elif not model_type == "universal" :
            for day in range(3) : # 3 days
                if model_type == "wind" :
                    wind_result = []
                    wind_result.append(hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 2, result[day]))
                    print(f'{day + 1} day: singular result wind test:{scaler.inverse_transform(wind_result)[0][2]}')
                
                elif model_type == "lat" : 
                    lat_result.append(hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 0, result[day]))
                    print(f'{day + 1} day: singular result lat test:{scaler.inverse_transform(lat_result)[0][0]}')
                
                else:
                    model = "long"
                    long_result = []
                    long_result.append(hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 1, result[day]))
                    print(f'{day + 1} day: singular result long test:{scaler.inverse_transform(long_result)[0][1]}')
        else :
            print('Unknown type of model or not yet configured')

if __name__ == "__main__" :
    fire.Fire(run_live_inference)