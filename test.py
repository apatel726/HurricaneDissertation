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

# TODO: Pass contents to data_utils for data preparation/feature extraction
# create hurricane objects for different unique hurricanes
for storm in data.storm_id.unique() :
    # get the storm entries
    entries = data[data['storm_id'] == storm]
    
    # convert to hurricane object
    hurricane = Hurricane(storm, storm)
    for index, entry in entries.iterrows():
        hurricane.add_entry(entry[2:])
    # run inference on this object
    hurricanes = [{ 'entries' : [{
            'time' : time,
            'wind' : hurricane.entries[time]['max_wind'],
            'lat' : hurricane.entries[time]['lat'],
            'lon' : hurricane.entries[time]['long'],
            'pressure' : hurricane.entries[time]['min_pressure']
        } for time in hurricane.entries],
        'storm' : storm
    }]
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
    # generate inference dictionary
    inference = inference(config['base_directory'], config['model_file'], config['scaler_file'], hurricanes)
    # create plotting file, including KML and a PNG ouput with a track
    plotting_utils.process_results({'inference' : inference, 'track' : args.test})
    # create a CSV for the output
    pd.DataFrame.from_dict(inference).to_csv(f'results/inferences.csv')