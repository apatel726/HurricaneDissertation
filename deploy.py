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
import pprint
import pickle
import json
import csv
import hurricane_ai.plotting_utils
from typing import List, Dict
from hurricane_ai.ml.bd_lstm_td import BidrectionalLstmHurricaneModel
from ingest import *
from datetime import timedelta

def inference(base_directory: str, model_file: str, scaler_file: str, file_type = "test") -> None:
    """
    Pulls live storm data and runs single pass inference for every storm.
    :param base_directory: Path to directory containing serialized artifacts (e.g. models, scalers).
    :param model_file: Filename of the model file.
    :param scaler_file: Filename of the scaler file.
    :param file_type: String, either "test" or "live"
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
        
    # 5 (6hour) increment depending on how the dataframe is structured
    lag = 5

    # Initialize model
    model = BidrectionalLstmHurricaneModel((None, None), model_type , os.path.join(base_directory, scaler_file),
                                           model_path=os.path.join(base_directory, model_file))
    # Grab live storm data
    # logic for file type
    if file_type == "live" :
        storms = nhc()
    elif type(file_type) == type(list()) :
        storms = file_type
    else :
        print("Unrecognized file type")
        return
    pp = pprint.PrettyPrinter()
    pp.pprint(storms)
    results = dict()
    for storm in storms:
        print(f"Running inference for {storm['storm']}")
        # Build data frame with raw observations and derived features
        df = prep_hurricane_data(storm["entries"], 1)
        
        if (len(storm["entries"])) <= 5 : # 1 entry = 6 hours 
            print(f'{storm["storm"]} does not have enough data (minimum 5 days)')
            continue
        
        # Run inference on the given observations
        result = model.predict(df, lag)
        print('-------------------------------------')
        
        # Converts the scaled values  from the model and scaler chosen to real values       
        with open(os.path.join(base_directory, scaler_file), 'rb') as f:
          scaler = pickle.load(f)
        
        # Run inference based on type of model
        wind_results = []
        lat_results = []
        lon_results = []
        if model_type == "universal" :
            for increment in range(5) : #5 6 hour increments
                wind_index = 0
                lat_index = 1
                long_index = 2
                
                # wind prints the wind for the first 5 (6hour) increments with an input shape of 11 features
                wind_result = scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 2, result[increment][wind_index])])[0][2]
                print(f'{(increment+1) * 6} hours: universal result wind test:{wind_result}')
                wind_results.append(wind_result)
                
                # lat prints the wind for the first 5 (6hour) increments with an input shape of 11 features
                lat_result = scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 0, result[increment][lat_index])])[0][0]
                lat_results.append(lat_result)
                print(f'{(increment+1) * 6} hours: universal result lat test:{lat_result}')
                
                # long prints the wind for the first 5 (6hour) increments with an input shape of 11 features
                long_result = -1 * scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 1, result[increment][long_index])])[0][1]
                lon_results.append(long_result)
                print(f'{(increment+1) * 6} hours: universal result long test:{long_result}')
            
        elif not model_type == "universal" :
            for increment in range(5) : # 5 (6hour) increments
                if model_type == "wind" :
                    wind_result = scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 2, result[increment])])[0][2]
                    print(f'{(increment+1) * 6} hours: universal result wind test:{wind_result}')
                    wind_results.append(wind_result)
                elif model_type == "lat" : 
                    lat_result = scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 0, result[increment])])[0][0]
                    lat_results.append(lat_result)
                    print(f'{(increment+1) * 6} hours: universal result lat test:{lat_result}')
                else:
                    model = "long"
                    long_result = -1 * scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 1, result[increment])])[0][1]
                    lon_results.append(long_result)
                    print(f'{(increment+1) * 6} hours: universal result long test:{long_result}')
        else :
            print('Unknown type of model or not yet configured')
        
        results[storm['storm']] = {
            'name' : storm["storm"],
            'times' : [df['time'].iloc[-1] + timedelta(hours = (6 * (i + 1))) for i in range(lag)],
            'wind' : wind_results,
            'lat' : lat_results,
            'lon' : lon_results
        }
        pp.pprint(results)
    
    return results
if __name__ == "__main__" :
    fire.Fire(inference)