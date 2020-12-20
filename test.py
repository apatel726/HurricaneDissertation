import argparse
import datetime
import json
import os
import pandas as pd
import tensorflow as tf
from deploy import inference
from hurricane_ai.container.hurricane_data_container import HurricaneDataContainer
from hurricane_ai.container.hurricane_data_container import Hurricane
from hurricane_ai import plotting_utils

# Create arugment parser for command line interface
# https://docs.python.org/3/howto/argparse.html
parser = argparse.ArgumentParser()

# cli flags for input file
parser.add_argument('--config', help = 'The file where all the configuration parameters are located', default = None)
# cli flags for test file
parser.add_argument('--test', help = 'The test file in HURDAT format to evaluate the models', default = None)
# cli flags for storm name
parser.add_argument('--name', help = 'The storm name in the test file to run inference on', default = None)
# Read in arguements
args = parser.parse_args()

# read in config file
with open(args.config) as f :
    config = json.load(f)

# TODO: Read in test file from hurricanecontrainer.py
data_container = HurricaneDataContainer()
data = data_container._parse(args.test)
def parse_entries(entries, storm) :
    '''
    "entries" : array of dict # The data for the storm in the form,
                {
                    'time' : Datetime,
                    'wind' : Knots,
                    'lat' : Decimal Degrees,
                    'lon' : Decimal Degrees,
                    'pressure' : Barometric pressure (mb)
                }
    '''
    return [{ 'entries' : [{
            'time' : time,
            'wind' : hurricane.entries[time]['max_wind'],
            'lat' : hurricane.entries[time]['lat'],
            'lon' : hurricane.entries[time]['long'],
            'pressure' : hurricane.entries[time]['min_pressure']
        } for time in entries],
        'storm' : storm
    }]

def create_table(prediction, storm, delta = 6) : 
    '''
    Creates an output table meant for CSV export
    
    Args:
        prediction (list(dict)) : The predictions in dict of form,
        'storm_id' : {
            'name' : String,
            'times' : list(Datetime),
            'wind' : list(float),
            'lat' : list(float),
            'lon' : list(float)
        }
    '''
    results = []
    for index, time in enumerate(prediction['universal'][storm.id]['times']) : 
        time = time.replace(tzinfo=None)
        result = {
            'time' : time,
            'delta' : delta * (index + 1),
            'Mpredict_Wind' : prediction['universal'][storm.id]['wind'][index],
            'Mpredict_Lat' : prediction['universal'][storm.id]['lat'][index],
            'Mpredict_Lon' : prediction['universal'][storm.id]['lon'][index],
            'Upredict_Wind' : prediction['singular'][storm.id]['wind'][index],
            'Upredict_Lat' : prediction['singular'][storm.id]['lat'][index],
            'Upredict_Lon' : prediction['singular'][storm.id]['lon'][index]
        }
        if time in storm.entries.keys() :
            truth_entry = storm.entries[time]
            result.update({'WindTruth' : truth_entry['max_wind'],
                        'LatTruth' : truth_entry['lat'],
                        'LonTruth' : truth_entry['long'] * -1,
                        'Mdiff_Wind' : truth_entry['max_wind'] - prediction['universal'][storm.id]['wind'][index],
                        'Mdiff_Lat' : truth_entry['lat'] - prediction['universal'][storm.id]['lat'][index],
                        'Mdiff_Lon' : (truth_entry['long'] * -1) - prediction['universal'][storm.id]['lon'][index],
                        'Udiff_Wind' : truth_entry['max_wind'] - prediction['singular'][storm.id]['wind'][index],
                        'Udiff_Lat' : truth_entry['lat'] - prediction['singular'][storm.id]['lat'][index],
                        'Udiff_Lon' : (truth_entry['long'] * -1) - prediction['singular'][storm.id]['lon'][index]})
        else :
            result.update({'WindTruth' : 'N/A',
                        'LatTruth' : 'N/A',
                        'LonTruth' : 'N/A',
                        'Mdiff_Wind' : 'N/A',
                        'Mdiff_Lat' : 'N/A',
                        'Mdiff_Lon' : 'N/A',
                        'Udiff_Wind' : 'N/A',
                        'Udiff_Lat' : 'N/A',
                        'Udiff_Lon' : 'N/A'})
        results.append(result)
    return results

# TODO: Pass contents to data_utils for data preparation/feature extraction
# create hurricane objects for different unique hurricanes
for storm in data.storm_id.unique() :
    # get the storm entries
    entries = data[data['storm_id'] == storm]
    
    # convert to hurricane object
    hurricane = Hurricane(storm, storm)
    for index, entry in entries.iterrows():
        hurricane.add_entry(entry[2:]) # start at index 2 because of HURDAT2 format
    
    # check to see if we're running on all time steps
    if "all_timesteps" in config.keys() :
        buffer = 1 if config["all_timesteps"]['placeholders'] else 5 # buffer determines start and end index
        inferences = []
        tables = dict()
        os.mkdir(f"results/{storm}_gis_files") # make a directory for the images and kml
        for index in range(buffer, len(hurricane.entries))  :
            timestamp = [* hurricane.entries][index]
            prediction = {
                'universal' : inference(config['base_directory'],
                                   config['model_file'],
                                   config['scaler_file'],
                                   parse_entries({
                                       time : hurricane.entries[time] for time in [* hurricane.entries][ : index + 1]
                                   }, storm)),
                'singular' : inference(config['univariate']['base_directory'],
                                   None,
                                   None,
                                   parse_entries({
                                       time : hurricane.entries[time] for time in [* hurricane.entries][ : index + 1]
                                   }, storm)) if 'univariate' in config.keys() else None
            }
            # note that this clears the memory, without this line, there's a fatal memory leak
            tf.keras.backend.clear_session()
            
            # add results to appropriate data structures
            tables[timestamp] = create_table(prediction,hurricane)
            inferences.append(prediction)
            
            # create plotting file, including KML and a PNG ouput with a track
            plotting_utils.process_results({
                    'inference' : prediction['universal'],
                    'track' : args.test
                },
                postfix = f"{storm}_gis_files/universal_{timestamp.strftime('%Y_%m_%d_%H_%M')}")
            if prediction['singular'] :
                plotting_utils.process_results({
                    'inference' : prediction['singular'],
                    'track' : args.test
                },
                postfix = f"{storm}_gis_files/singular_{timestamp.strftime('%Y_%m_%d_%H_%M')}")
        
        # Save to excel sheet
        print("Writing files to Excel . . . ", end = '')
        with pd.ExcelWriter(f"results/{storm}.xlsx") as writer :
            for timestep in tables :
                pd.DataFrame.from_dict(tables[timestep]).to_excel(
                    writer, sheet_name = timestep.strftime("%Y_%m_%d_%H_%M"))
        print("Done!")
        
    else :
        # generate inference dictionary
        inferences = {
            'universal' : inference(config['base_directory'],
                               config['model_file'],
                               config['scaler_file'],
                               parse_entries(hurricane.entries, storm)),
            'singular' : inference(config['univariate']['base_directory'],
                               None,
                               config['univariate']['scaler_file'],
                               parse_entries(hurricane.entries, storm)) if 'univariate' in config.keys() else None
        }
        # create plotting file, including KML and a PNG ouput with a track
        plotting_utils.process_results({'inference' : inferences['universal'], 'track' : args.test}, postfix = 'universal')
        if inferences['singular'] :
            plotting_utils.process_results({'inference' : inferences['singular'], 'track' : args.test}, postfix = 'singular')
        # create a CSV for the output
        pd.DataFrame.from_dict(create_table(inferences,hurricane)
                              ).to_csv(f'results/inferences_{[* hurricane.entries][-1].strftime("%Y_%m_%d_%H_%M")}.csv')