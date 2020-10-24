# Akash Patel
# PURPOSE: To provide a class and API for the Official NHC Atlantic Track and
# Intensity Forecast Errors (1970-2016, excluding depressions)
# METHOD: Read the text file provided by the NOAA and NHC into a pandas DataFrame and
# create an API for common functions and applications
# OUTPUT: Forecast Error Database http://www.nhc.noaa.gov/verification/verify7.shtml

import datetime
import pickle as pkl
from datetime import timedelta
from os import path
import os

from hurricane_ai import ERROR_SOURCE_FILE, ERROR_PKL_FILE, is_source_modified


class ErrorModel:
    """
    PURPOSE: To create a class for each model included in the forecast error database
    METHOD: Provide an API
    OUTPUT: A class with a DataFrame and associated operations
    """

    def __init__(self, model_name: str):
        """
        Sets model name and instantiates new storm data dictionary for the given NHC model name.
        :param model_name: The name of the NHC model.
        """
        self.name = model_name
        self.storms = dict()
        return

    def add_entry(self, storm_id, timestamp, **kwargs):
        """
        Adds a new measurement for a given storm at a given time.
        :param storm_id: The ID of the storm to which the measurement corresponds.
        :param timestamp: The time at which the measurement was taken.
        :param kwargs: Measurement data.
        """

        # Add new entry for the given storm ID if it doesn't already exist
        if storm_id not in self.storms.keys():
            self.storms[storm_id] = dict()

        # Add observation for the given storm
        self.storms[storm_id].update({
            timestamp: {
                "sample_sizes": kwargs['sample_sizes'],
                "lat": kwargs['lat'],
                "lon": kwargs['lon'],
                "wind_speed": kwargs['wind_speed'],
                "intensity_forecast": kwargs['int_fcst'],
                "track_forecast": kwargs['trk_fcst'],
            }
        })


class ErrorModelContainer:
    """
    Encapsulates storm error model measurements.
    """

    def __init__(self):
        """
        Reads in the text file with NHC model errors.
        :param source_file: Official forecast errors
        :param pkl_file: Pickle file containing post-processed forecast error objects
        """

        # Read in the previously-serialized error data if it exists and the source data hasn't been modified
        if path.exists(ERROR_PKL_FILE) and not is_source_modified(ERROR_SOURCE_FILE, ERROR_PKL_FILE):
            with open(ERROR_PKL_FILE, 'rb') as in_file:
                self.error_models = pkl.load(in_file)
            return

        # Parse file and construct dictionary of error model object
        self.error_models = self._parse_from_raw(ERROR_SOURCE_FILE)

        # Serialize error models to file
        os.makedirs(os.path.dirname(ERROR_PKL_FILE), exist_ok=True)
        with open(ERROR_PKL_FILE, 'wb') as out_file:
            pkl.dump(self.error_models, out_file)

    @staticmethod
    def _parse_from_raw(filename: str) -> dict:
        """
        PURPOSE: Parse in the Forecast Error database
        METHOD: Use the forecast error file format provided by the NHC and NOAA and load into model dictionary
        OUTPUT: pandas DataFrame with appropriate container
        REFERENCES:
        [1] http://www.nhc.noaa.gov/verification/errors/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt
        [2] http://www.nhc.noaa.gov/verification/pdfs/Error_Tabulation_File_Format.pdf
        :param filename: Official forecast errors
        """

        error_models = dict()

        # Begin parsing by reading in line by line
        with open(filename) as raw:
            lines = raw.readlines()

            # Get model names and declare model objects
            line = lines[1].split()
            model_names = line[2:]
            for model_name in model_names:
                error_models[model_name] = ErrorModel(model_name)

            # Data starts at line 9
            for line in lines[9:]:
                line = line.split()

                # Identify atlantic storm date, storm id, associated sample sizes, latitude and longitude, and windspeed
                timestamp = datetime.datetime.strptime(line[0], "%d-%m-%Y/%H:%M:%S")
                storm_id = line[1]
                sample_sizes = {"F012": float(line[2]), "F024": float(line[3]), "F036": float(line[4]),
                                "F048": float(line[5]), "F072": float(line[6]), "F096": float(line[7]),
                                "F120": float(line[8]), "F144": float(line[9]), "F168": float(line[10])}
                latitude = float(line[11])
                longitude = float(line[12])
                wind_speed = int(line[13])

                # Iterate through model forecast track and intensity errors
                for i in range(len(model_names)):
                    intensity_forecast = dict(list(zip(
                        [timestamp, timestamp + timedelta(hours=12), timestamp + timedelta(hours=24),
                         timestamp + timedelta(hours=36), timestamp + timedelta(hours=48),
                         timestamp + timedelta(hours=72), timestamp + timedelta(hours=96),
                         timestamp + timedelta(hours=120), timestamp + timedelta(hours=144),
                         timestamp + timedelta(hours=168)],
                        [None if x == "-9999.0" else float(x) for x in line[14 + (20 * i): 24 + (20 * i)]])))
                    track_forecast = dict(list(zip(
                        [timestamp, timestamp + timedelta(hours=12), timestamp + timedelta(hours=24),
                         timestamp + timedelta(hours=36), timestamp + timedelta(hours=48),
                         timestamp + timedelta(hours=72), timestamp + timedelta(hours=96),
                         timestamp + timedelta(hours=120), timestamp + timedelta(hours=144),
                         timestamp + timedelta(hours=168)],
                        [None if x == "-9999.0" else float(x) for x in line[24 + (20 * i): 34 + (20 * i)]])))

                    # Add forecast to model and storm, initialize if storm id does not exist
                    error_models[model_names[i]].add_entry(storm_id, timestamp, sample_sizes=sample_sizes, lat=latitude,
                                                           lon=longitude, wind_speed=wind_speed,
                                                           int_fcst=intensity_forecast, trk_fcst=track_forecast)

        return error_models
