# Akash Patel
# Last Update: 3/25/30
# Read the text file from NOAA to convert into pandas DataFrame
# format from http://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atlantic.pdf

import csv
import datetime
import pickle as pkl
from os import path
import os

import pandas as pd

from hurricane_ai import HURRICANE_SOURCE_FILE, HURRICANE_PKL_FILE, HURRICANE_IDS_FILE, \
    is_source_modified


class Hurricane(object):

    def __init__(self, hurricane_name, hurricane_id):
        """
        Initialize Hurricane object
        :param hurricane_name: Human-readable hurricane name
        :param hurricane_id: Unique hurricane identifier
        """

        # Set instance variables
        self.name = hurricane_name
        self.id = hurricane_id
        self.entries = dict()
        self.models = dict()

    def add_entry(self, array):
        """
        Adds a hurricane track entry based on standard HURDAT2 format
        :param array: Hurricane track entry to add
        """

        self.entries.update({
            array[0]: {  # dateteime of entry
                'entry_time': array[0],
                'entry_id': array[1],
                'entry_status': array[2],
                'lat': float(array[3][:-1]),  # Convert to number from format '#.#N'
                'long': float(array[4][:-1]),  # Convert to number from format '#.#W'
                'max_wind': float(array[5]),
                'min_pressure': None if array[6] is None else float(array[6]),  # Early records are -999 or None
                'wind_radii': array[7:],  # Array based on HURDAT2 format
            }
        })

    def add_model(self, name, model):
        """
        Adds hurricane model errors
        :param name: Human-readable hurricane name
        :param model: The hurricane model
        """

        self.models[name] = model


class HurricaneDataContainer:
    # Constants for validation when parsing
    STORM_HEADER_ROW_LENGTH = 3
    STORM_ENTRY_ROW_LENGTH = 20
    MISSING_PRESSURE_SYM = "-999"

    def __init__(self):
        """
        Initializes the hurricane container API
        :param filename: Hurricane measurement data file.
        :param pkl_file: Pickle file containing post-processed hurricane measurement data.
        :param storm_id_file: Text file containing list of unique hurricane IDs.
        """

        # Read in the previously-serialized hurricane data if it exists and the source data hasn't been modified
        if path.exists(HURRICANE_PKL_FILE) and not is_source_modified(HURRICANE_SOURCE_FILE,
                                                                      HURRICANE_PKL_FILE) and path.exists(
            HURRICANE_IDS_FILE):

            self.storm_ids = []

            # Read hurricane data
            with open(HURRICANE_PKL_FILE, 'rb') as in_file:
                self.hurricanes = pkl.load(in_file)

            # Read hurricane IDs
            with open(HURRICANE_IDS_FILE, 'r') as in_file:
                for line in in_file:
                    # Everything up to the newline character
                    self.storm_ids.append(line[:-1])

            return

        # Initialize hurricanes and storms ID dictionaries
        self.storm_ids, self.hurricanes = self._get_hurricanes_ids(self._parse(HURRICANE_SOURCE_FILE))

        # Serialize hurricane measurements to file
        os.makedirs(os.path.dirname(HURRICANE_PKL_FILE), exist_ok=True)
        with open(HURRICANE_PKL_FILE, 'wb') as out_file:
            pkl.dump(self.hurricanes, out_file)

        # Serialize hurricane IDs to file
        with open(HURRICANE_IDS_FILE, 'w') as out_file:
            for storm_id in self.storm_ids:
                out_file.write('{}\n'.format(storm_id))

    def __iter__(self):
        """
        Generates hurricane objects for iteration.

        :return: Next hurricane object in the generator sequence.
        """
        for hurricane in self.hurricanes.values():
            yield hurricane

    def __len__(self):
        """
        Gets the number of hurricanes in the container.

        :return: The number of hurricanes in the container.
        """
        return len(self.hurricanes)

    @staticmethod
    def _get_hurricanes_ids(hurricanes_df: pd.DataFrame) -> (dict, dict):
        """
        Converts hurricanes container frame into dictionary of hurricane classes and dictionary of unique storm IDs.
        :return: Type of storm IDs and hurricane objects
        """

        storm_ids = dict()
        hurricanes = dict()

        for index, entry in hurricanes_df.iterrows():
            if entry['storm_id'] not in hurricanes:
                hurricanes[entry['storm_id']] = Hurricane(entry['storm_name'], entry['storm_id'])
                storm_ids[entry['storm_id']] = entry['storm_name']
                # Add entry to hurricane
            hurricanes[entry['storm_id']].add_entry(entry[2:])

        return storm_ids, hurricanes

    def _parse(self, filename: str) -> pd.DataFrame:
        """
        Parses hurricane container and stores in class member container frame.
        :param filename: Default filename.
        :return: Instantiated hurricanes container frame.
        """

        db = []

        with open(filename) as raw:
            csv_reader = csv.reader(raw, delimiter=',', quoting=csv.QUOTE_NONE, skipinitialspace=True)

            for storm_row in csv_reader:

                # Remove last entry resulting from trailing comma
                storm_row = storm_row[:-1]

                assert len(storm_row) == self.STORM_HEADER_ROW_LENGTH, \
                    'Invalid storm header format - expected {} columns, encountered {}'.format(
                        self.STORM_HEADER_ROW_LENGTH, len(storm_row))

                # Extract current storm details
                storm_id = storm_row[0]
                storm_name = storm_row[1]
                storm_entries = storm_row[2]

                # Skip non-Atlantic storms
                if storm_row[0][:2] != 'AL':

                    print("Error, unidentified storm {}".format(storm_row[0]))

                    # Advance by the specified number of entries
                    for _ in range(int(storm_entries)):
                        next(csv_reader)

                    continue

                for _ in range(int(storm_entries)):
                    # Read in next row and remove last column (handles trailing comma)
                    entry_row = next(csv_reader)[:-1]

                    assert len(entry_row) == self.STORM_ENTRY_ROW_LENGTH, \
                        'Invalid storm entry format - expected {} columns, encountered {}'.format(
                            self.STORM_ENTRY_ROW_LENGTH, len(entry_row))

                    # Set missing pressures (represented by -999) to None
                    entry = [None if x == self.MISSING_PRESSURE_SYM else x for x in entry_row]

                    # Construct date and time entries based on columns
                    timestamp = datetime.datetime(int(entry[0][:4]),
                                                  int(entry[0][4:6]),
                                                  int(entry[0][6:8]),
                                                  int(entry[1][:2]),
                                                  int(entry[1][3:]))

                    # Concatenate storm details and current timestamp, track and pressure
                    db.append([storm_id, storm_name, timestamp] + entry[2:])

        return pd.DataFrame(db,
                            columns=['storm_id', 'storm_name', 'entry_time', 'entry_id', 'entry_status', 'lat', 'long',
                                     'max_wind', 'min_pressure', '34kt_ne', '34kt_se', '34kt_sw', '34kt_nw', '50kt_ne',
                                     '50kt_se', '50kt_sw', '50kt_nw', '64kt_ne', '64kt_se', '64kt_sw', '64kt_nw'])
