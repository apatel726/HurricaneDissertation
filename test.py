import argparse
import datetime
import json
import pandas as pd
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
        for index in range(buffer, len(hurricane.entries))  :
            inferences.append(inference(config['base_directory'],
                                   config['model_file'],
                                   config['scaler_file'],
                                   parse_entries({time : hurricane.entries[time] for time in [* hurricane.entries][ : index + 1]}, storm)))
            # create plotting file, including KML and a PNG ouput with a track
            plotting_utils.process_results({
                'inference' : inferences[-1],
                'track' : args.test
            },
            postfix = f"_{[* hurricane.entries][index].strftime('%Y_%m_%d_%H_%M')}")
        # save to csv
        pd.DataFrame.from_dict(inferences).to_csv(f'results/inferences_{datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M")}.csv')
    else :
        # generate inference dictionary
        inferences = inference(config['base_directory'], config['model_file'], config['scaler_file'], parse_entries(hurricane.entires, storm))
        # create plotting file, including KML and a PNG ouput with a track
        plotting_utils.process_results({'inference' : inferences, 'track' : args.test})
        # create a CSV for the output
        pd.DataFrame.from_dict(inferences).to_csv(f'results/inferences_{datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M")}.csv')