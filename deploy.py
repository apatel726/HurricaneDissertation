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
import xmltodict
import fire
import requests
from datetime import datetime
import dateutil.parser
from pytz import timezone
import zipfile
import io
import pandas as pd
import pickle
import json
import hurricane_ai.plotting_utils
from typing import List, Dict

from hurricane_ai.ml.bd_lstm_td import BidrectionalLstmHurricaneModel

def past_track(link):
    '''
    From a KMZ file of a storm in the NHC format, we extract the history
    Parameters
    ----------
    link string
        The network link or downloadable KMZ href file
    Returns
    -------
    dict
    '''
    kmz = requests.get(link)
    uncompressed = zipfile.ZipFile(io.BytesIO(kmz.content))

    # get the kml name
    for name in uncompressed.namelist():
        # all kml file names begin with al, e.g. 'al202020.kml'
        if name[:2] == 'al':
            file_name = name

    # read the contents of the kml file in the archive
    kml = xmltodict.parse(uncompressed.read(file_name))
    kml['results'] = []
    for attribute in kml['kml']['Document']['Folder']:
        if attribute['name'] == 'Data':
            for entry in attribute['Placemark']:
                # parse time information
                time = datetime.strptime(entry['atcfdtg'],
                                        '%Y%m%d%H').replace(
                    tzinfo=timezone('UTC'))

                # add to results
                kml['results'].append({
                    'time' : time,
                    'wind' : float(entry['intensity']),
                    'lat' : float(entry['lat']),
                    'lon' : float(entry['lon']),
                    'pressure' : float(entry['minSeaLevelPres'])
                })
                print(kml['results'][-1])

    return kml

def nhc() -> List[Dict[str, List]]:
    '''
    Runs the NHC update and populates current Atlantic storms
    Returns
    -------
    array of dict
        Each dictionary is in the following form,
        {
            "storm" : string # the storm ID from the NHC
            "metadata" : dict # the kml files used to create the results
            "entries" : array of dict # The data for the storm in the form,
                {
                    'time' : Datetime,
                    'wind' : Knots,
                    'lat' : Decimal Degrees,
                    'lon' : Decimal Degrees,
                    'pressure' : Barometric pressure (mb)
                }
        }
    '''
    # this link can be reused to download the most recent data
    static_link = 'https://www.nhc.noaa.gov/gis/kml/nhc_active.kml'
    # common timezones for parsing with dateutil. offset by seconds
    timezones = {
        "ADT": 4 * 3600,
        "AST": 3 * 3600,
        "CDT": -5 * 3600,
        "CST": -6 * 3600,
        "CT": -6 * 3600,
        "EDT": -4 * 3600,
        "EST": -5 * 3600,
        "ET": -5 * 3600,
        "GMT": 0 * 3600,
        "PST": -8 * 3600,
        "PT": -8 * 3600,
        "UTC": 0 * 3600,
        "Z": 0 * 3600,
    }

    # create data structure as dictionary
    request = requests.get(static_link)
    data = xmltodict.parse(request.text)
    results = []
    
    # return if no storms
    if 'Folder' not in data['kml']['Document'].keys() :
        return
    
    # parse in storms
    for folder in data['kml']['Document']['Folder']:
        # the id's that start with 'at' are the storms we are interested in
        # others can include 'wsp' for wind speed probabilities
        if folder['@id'][:2] == 'at':
            # some storms don't have any data because they are so weak
            if not 'ExtendedData' in folder.keys():
                continue

            # storm data structure
            storm = {
                'metadata': folder,
                'entries': []
            }
            entry = {}

            for attribute in folder['ExtendedData'][1]:
                if attribute == 'tc:atcfID':  # NHC Storm ID
                    storm['id'] = folder['ExtendedData'][1][attribute]
                elif attribute == 'tc:name':  # Human readable name
                    print(folder['ExtendedData'][1][attribute])
                elif attribute == 'tc:centerLat':  # Latitude
                    entry['lat'] = float(folder['ExtendedData'][1][attribute])
                elif attribute == 'tc:centerLon':  # Longitude
                    entry['lon'] = float(folder['ExtendedData'][1][attribute])
                elif attribute == 'tc:dateTime':  # Timestamp
                    entry['time'] = dateutil.parser.parse(
                        folder['ExtendedData'][1][attribute],
                        tzinfos=timezones)
                elif attribute == 'tc:minimumPressure':  # Barometric pressure
                    entry['pressure'] = float(folder['ExtendedData'][1]
                                              [attribute].split(' ')[0])
                elif attribute == 'tc:maxSustainedWind':  # Wind Intensity
                    # note that we are converting mph to knots
                    entry['wind'] = float(folder['ExtendedData'][1][attribute].
                                          split(' ')[0]) / 1.151

            print(storm['id'])
            print(entry)

            # add entry to storm
            storm['entries'].append(entry)
            # get network link and extract past history
            for links in folder['NetworkLink']:
                if links['@id'] == 'pasttrack':
                    kml = past_track(links['Link']['href'])
                    # add history to entries
                    storm['entries'].extend(kml['results'])

                    # add history to storm metadata
                    storm['metadata']['history'] = kml

            # add to results
            results.append(storm)

    return results


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

    # 5 day lag
    lag = 5

    # Initialize model
    model = BidrectionalLstmHurricaneModel((None, None), "wind", os.path.join(base_directory, scaler_file),
                                           model_path=os.path.join(base_directory, model_file))

    # Grab live storm data
    live_storms = nhc()

    for storm in live_storms:
        print(f"Running inference for {storm['metadata']['name']}")

        # Build data frame with raw observations and derived features
        df = prep_hurricane_data(storm["entries"], lag)
       # df2 = df.iloc[[0, -1]]
        
        if (len(storm["entries"])) <= 20 : # 1 entry = 6 hours, 20 entries is 120 hours (5 days)
            print(f'{storm["metadata"]["name"]} does not have enough data (minimum 5 days)')
            continue
        
        # load current model configuration
        with open(os.path.join(base_directory, 'hyperparameters.json')) as f:
            root = json.load(f)
        
        # Run inference on the given observations
        result = model.predict(df, lag)
        print('-------------------------------------')
        
 #Converts the scaled values  from the model and scaler chosen to real values       
        with open(os.path.join(base_directory, scaler_file), 'rb') as f:
          scaler = pickle.load(f)
        
        # Run inference based on type of model
        dictimport = root["config"]  
        if root['universal'] :
            # Print result
            
            for day in range(3) : # 3 days
                wind_index = 0
                lat_index = 1
                long_index = 2
                
                # wind prints the wind for the first 3 days with an input shape of 11 features
                wind_result = []
                print(result.shape)
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
            
        elif not root['universal'] :
            featuresearch = dictimport["name"]
            
            for day in range(3) : # 3 days
                if featuresearch == "sequential":
                    model = "wind"
                    wind_result = []
                    wind_result.append(hurricane_ai.plotting_utils._generate_sparse_feature_vector(11, 2, result[day]))
                    print(f'{day + 1} day: singular result wind test:{scaler.inverse_transform(wind_result)[0][2]}')
                
                elif featuresearch == 'sequential_1':
                    model = "lat"
                    lat_result = []
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