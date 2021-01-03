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
import math
import pandas as pd
import numpy as np
import pprint
import pickle
import json
import csv
import hurricane_ai.plotting_utils
from typing import List, Dict
from hurricane_ai.ml.bd_lstm_td import BidrectionalLstmHurricaneModel
from ingest import *
from datetime import timedelta

def inference(base_directory: str, model_file: str, scaler_file: str, output_times: list, file_type = "test") -> None :
    """
    Pulls live storm data and runs single pass inference for every storm.
    :param base_directory: Path to directory containing serialized artifacts (e.g. models, scalers).
    :param model_file: Filename of the model file.
    :param scaler_file: Filename of the scaler file.
    :param file_type: String or a list of entry objects. String can be value "live". List of dict objects
        in the form, 
        [{ 'entries' : [{
            'time' : Datetime,
            'wind' : float,
            'lat' : float,
            'lon' : float,
            'pressure' : float
        } for time in entries],
        'storm' : storm
        }]
    """
    # load current model configuration
    with open(os.path.join(base_directory, 'hyperparameters.json')) as f:
        root = json.load(f)
        if root['universal'] :
            model_type = "universal"
            # Initialize model
            model = BidrectionalLstmHurricaneModel((None, None), model_type , os.path.join(base_directory, scaler_file),
                                                   model_path=os.path.join(base_directory, model_file))
        elif root['singular']:
            model_type = "singular"
            # Initialize all models
            directories = os.listdir(base_directory)
            model = dict()
            for directory in directories :
                if directory[:4] == 'wind' :
                    model['wind'] = BidrectionalLstmHurricaneModel((None, None), 'wind',
                                                                   os.path.join(base_directory + f"/{directory}",
                                                                                'feature_scaler.pkl'),
                                                                   model_path=os.path.join(base_directory,
                                                                                           directory + f"/model{directory[4:]}"))
                elif directory[:3] == 'lat' :
                    model['lat'] = BidrectionalLstmHurricaneModel((None, None), 'lat',
                                                                   os.path.join(base_directory + f"/{directory}",
                                                                                'feature_scaler.pkl'),
                                                                   model_path=os.path.join(base_directory,
                                                                                           directory + f"/model{directory[3:]}"))
                elif directory[:3] == 'lon' :
                    model['lon'] = BidrectionalLstmHurricaneModel((None, None), 'lon',
                                                                   os.path.join(base_directory + f"/{directory}",
                                                                                'feature_scaler.pkl'),
                                                                   model_path=os.path.join(base_directory,
                                                                                           directory + f"/model{directory[3:]}"))
        
    # 5 (6hour) increment depending on how the dataframe is structured
    lag = 5
    
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
        
        # Run inference based on type of model
        wind_results = []
        lat_results = []
        lon_results = []
        if model_type == "universal" :
            # Run inference on the given observations
            result = model.predict(df, lag)
            
            for increment in range(5) : #5 6 hour increments
                wind_index = 0
                lat_index = 1
                lon_index = 2
                
                # wind prints the wind for the first 5 (6hour) increments with an input shape of 11 features
                wind_result = model.scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 2, result[increment][wind_index])])[0][2]
                print(f'{output_times[increment]} hours: universal result wind test:{wind_result}')
                wind_results.append(wind_result)
                
                # lat prints the wind for the first 5 (6hour) increments with an input shape of 11 features
                lat_result = model.scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 0, result[increment][lat_index])])[0][0]
                lat_results.append(lat_result)
                print(f'{output_times[increment]} hours: universal result lat test:{lat_result}')
                
                # lon prints the wind for the first 5 (6hour) increments with an input shape of 11 features
                lon_result = -1 * model.scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 1, result[increment][lon_index])])[0][1]
                lon_results.append(lon_result)
                print(f'{output_times[increment]} hours: universal result lon test:{lon_result}')
            
        elif model_type == "singular" :
            # Run inference on the given observations
            result = {
                'wind' : model['wind'].predict(df, lag),
                'lat' : model['lat'].predict(df, lag),
                'lon' : model['lon'].predict(df, lag)
            }
            for increment in range(5) : # 5 (6hour) increments
                wind_result = model['wind'].scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 2, result['wind'][increment])])[0][2]
                wind_results.append(wind_result)
                print(f'{output_times[increment]} hours: singular result wind test:{wind_result}')
                
                lat_result = model['lat'].scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 0, result['lat'][increment])])[0][0]
                lat_results.append(lat_result)
                print(f'{output_times[increment]} hours: singular result lat test:{lat_result}')
                
                lon_result = -1 * model['lon'].scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 1, result['lon'][increment])])[0][1]
                lon_results.append(lon_result)
                print(f'{output_times[increment]} hours: singular result lon test:{lon_result}')
        else :
            print('Unknown type of model or not yet configured')
        
        results[storm['storm']] = {
            'name' : storm["storm"],
            'times' : [df['time'].iloc[-1] + timedelta(hours = time) for time in output_times],
            'wind' : wind_results,
            'lat' : lat_results,
            'lon' : lon_results
        }
        pp.pprint(results)
    
    del model
    
    return results

def batch_inference(base_directory: str, model_file: str, scaler_file: str, output_times: list, storms: dict) -> None :
    """
    Pulls live storm data and runs single pass inference for every storm.
    :param base_directory: Path to directory containing serialized artifacts (e.g. models, scalers).
    :param model_file: Filename of the model file.
    :param scaler_file: Filename of the scaler file.
    :param file_type: String or a list of entry objects. String can be value "live". List of dict objects
        in the form, 
        [{  'storm' : storm,
            'entries' : [ {
                'time' : Datetime,
                'wind' : float,
                'lat' : float,
                'lon' : float,
                'pressure' : float
            } for time in entries ]
         }]
    """
    # load current model configuration
    with open(os.path.join(base_directory, 'hyperparameters.json')) as f:
        root = json.load(f)
        if root['universal'] :
            model_type = "universal"
            # Initialize model
            model = BidrectionalLstmHurricaneModel((None, None), model_type , os.path.join(base_directory, scaler_file),
                                                   model_path=os.path.join(base_directory, model_file))
        elif root['singular']:
            model_type = "singular"
            # Initialize all models
            directories = os.listdir(base_directory)
            model = dict()
            for directory in directories :
                if directory[:4] == 'wind' :
                    model['wind'] = BidrectionalLstmHurricaneModel((None, None), 'wind',
                                                                   os.path.join(base_directory + f"/{directory}",
                                                                                'feature_scaler.pkl'),
                                                                   model_path=os.path.join(base_directory,
                                                                                           directory + f"/model{directory[4:]}"))
                elif directory[:3] == 'lat' :
                    model['lat'] = BidrectionalLstmHurricaneModel((None, None), 'lat',
                                                                   os.path.join(base_directory + f"/{directory}",
                                                                                'feature_scaler.pkl'),
                                                                   model_path=os.path.join(base_directory,
                                                                                           directory + f"/model{directory[3:]}"))
                elif directory[:3] == 'lon' :
                    model['lon'] = BidrectionalLstmHurricaneModel((None, None), 'lon',
                                                                   os.path.join(base_directory + f"/{directory}",
                                                                                'feature_scaler.pkl'),
                                                                   model_path=os.path.join(base_directory,
                                                                                           directory + f"/model{directory[3:]}"))
    
    lag = 5 # 5 (6hour) increments depending on how the dataframe is structured
    wind_index = 0
    lat_index = 1
    lon_index = 2
    batch_size = 32
    
    pp = pprint.PrettyPrinter()
    pp.pprint(storms)
    results = dict()
    print("Creating batch file")
    for storm in storms :
        # create predictions        
        if model_type == "universal" :
            # create results from batches
            raw_results = np.asarray([model.model.predict(batch_inputs)
                           for batch_inputs in np.array_split([model.scaler.transform(prep_hurricane_data(inputs, 1)[model.FEATURES].tail(lag).values)
                                                               for inputs in [storm['entries'][i : i + lag + 1]
                                                                              for i in range(len(storm['entries']) - lag)]],
                                                              math.ceil((len(storm['entries']) - lag) / batch_size))])
            
            # translate predictions and add to results. we combine the batches here
            results[storm['storm']] = { storm['entries'][index + lag]['time'] : {
                'wind' : [inverse_scaled[2] for inverse_scaled in model.scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 2, result[i][wind_index])
                     for i in range(lag)])],
                'lat' : [inverse_scaled[0] for inverse_scaled in model.scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 0, result[i][lat_index])
                     for i in range(lag)])],
                'lon' : [inverse_scaled[1] for inverse_scaled in model.scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 1, result[i][lon_index])
                     for i in range(lag)])]
            } for index, result in enumerate([result for sublist in raw_results for result in sublist])}
            
            pp.pprint(results[storm['storm']])
        
        elif model_type == "singular" :
            raw_results = {
                'wind' : np.asarray([model['wind'].model.predict(batch_inputs)
                                     for batch_inputs in np.array_split([model['wind'].scaler.transform(
                                         prep_hurricane_data(inputs, 1)[model['wind'].FEATURES].tail(lag).values)
                      for inputs in [storm['entries'][i : i + lag + 1] for i in range(len(storm['entries']) - lag)]],
                                                              math.ceil((len(storm['entries']) - lag) / batch_size))]),
                'lat' : np.asarray([model['lat'].model.predict(batch_inputs)
                                     for batch_inputs in np.array_split([model['lat'].scaler.transform(
                                         prep_hurricane_data(inputs, 1)[model['lat'].FEATURES].tail(lag).values)
                      for inputs in [storm['entries'][i : i + lag + 1] for i in range(len(storm['entries']) - lag)]],
                                                              math.ceil((len(storm['entries']) - lag) / batch_size))]),
                'lon' : np.asarray([model['lon'].model.predict(batch_inputs)
                                     for batch_inputs in np.array_split([model['lon'].scaler.transform(
                                         prep_hurricane_data(inputs, 1)[model['lon'].FEATURES].tail(lag).values)
                      for inputs in [storm['entries'][i : i + lag + 1] for i in range(len(storm['entries']) - lag)]],
                                                              math.ceil((len(storm['entries']) - lag) / batch_size))])
            }
            
            # reshape batch files
            raw_results['wind'] = [result for sublist in raw_results['wind'] for result in sublist]
            raw_results['lat'] = [result for sublist in raw_results['lat'] for result in sublist]
            raw_results['lon'] = [result for sublist in raw_results['lon'] for result in sublist]
            
            # translate predictions and add to results
            results[storm['storm']] = { storm['entries'][i + lag]['time'] : {
                'wind' : [inverse_scaled[2] for inverse_scaled in model['wind'].scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 2, raw_results['wind'][i][j])
                     for j in range(lag)])],
                'lat' : [inverse_scaled[0] for inverse_scaled in model['lat'].scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 0, raw_results['lat'][i][j])
                     for j in range(lag)])],
                'lon' : [inverse_scaled[1] for inverse_scaled in model['lon'].scaler.inverse_transform(
                    [hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 1, raw_results['lon'][i][j])
                     for j in range(lag)])]
            } for i in range(len(raw_results['wind']))} # length for wind, lat, and lon are the same
            
            pp.pprint(results[storm['storm']])
    
    return results

if __name__ == "__main__" :
    fire.Fire(inference)