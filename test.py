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

def create_table(prediction, storm) : 
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
    for index, time in enumerate(prediction[storm.id]['times']) : 
        time = time.replace(tzinfo=None)
        if time in storm.entries.keys() :
            truth_entry = storm.entries[time]
            results.append({'WindTruth' : truth_entry['max_wind'],
                        'LatTruth' : truth_entry['lat'],
                        'LonTruth' : truth_entry['long'] * -1,
                        'Mpredict_Wind' : prediction[storm.id]['wind'][index],
                        'Mpredict_Lat' : prediction[storm.id]['lat'][index],
                        'Mpredict_Lon' : prediction[storm.id]['lon'][index],
                        'Upredict_Wind' : 'TODO',
                        'Upredict_Lat' : 'TODO',
                        'Upredict_Lon' : 'TODO',
                        'Mdiff_Wind' : truth_entry['max_wind'] - prediction[storm.id]['wind'][index],
                        'Mdiff_Lat' : truth_entry['lat'] - prediction[storm.id]['lat'][index],
                        'Mdiff_Lon' : (truth_entry['long'] * -1) - prediction[storm.id]['lon'][index],
                        'Udiff_Wind' : 'TODO',
                        'Udiff_Lat' : 'TODO',
                        'Udiff_Lon' : 'TODO'})
        else :
            results.append({'WindTruth' : 'N/A',
                        'LatTruth' : 'N/A',
                        'LonTruth' : 'N/A',
                        'Mpredict_Wind' : prediction[storm.id]['wind'][index],
                        'Mpredict_Lat' : prediction[storm.id]['lat'][index],
                        'Mpredict_Lon' : prediction[storm.id]['lon'][index],
                        'Upredict_Wind' : 'TODO',
                        'Upredict_Lat' : 'TODO',
                        'Upredict_Lon' : 'TODO',
                        'Mdiff_Wind' : 'N/A',
                        'Mdiff_Lat' : 'N/A',
                        'Mdiff_Lon' : 'N/A',
                        'Udiff_Wind' : 'TODO',
                        'Udiff_Lat' : 'TODO',
                        'Udiff_Lon' : 'TODO'})
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
        for index in range(buffer, len(hurricane.entries))  :
            prediction = inference(config['base_directory'],
                                   config['model_file'],
                                   config['scaler_file'],
                                   parse_entries({time : hurricane.entries[time] for time in [* hurricane.entries][ : index + 1]}, storm))
            inferences.append(prediction)
            # create plotting file, including KML and a PNG ouput with a track
            plotting_utils.process_results({
                'inference' : prediction,
                'track' : args.test
            },
            postfix = f"_{[* hurricane.entries][index].strftime('%Y_%m_%d_%H_%M')}")
            # save to csv
            pd.DataFrame.from_dict(create_table(prediction,hurricane)
                                  ).to_csv(f"results/inferences_{storm}_{[* hurricane.entries][index].strftime('%Y_%m_%d_%H_%M')}.csv")
    else :
        # generate inference dictionary
        inferences = inference(config['base_directory'], config['model_file'], config['scaler_file'], parse_entries(hurricane.entries, storm))
        # create plotting file, including KML and a PNG ouput with a track
        plotting_utils.process_results({'inference' : inferences, 'track' : args.test})
        # create a CSV for the output
        pd.DataFrame.from_dict(create_table(inferences,hurricane)).to_csv(f'results/inferences_{[* hurricane.entries][-1].strftime("%Y_%m_%d_%H_%M")}.csv')