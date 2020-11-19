# Akash Patel
# Last Update: 11/17/20
# Read the text file from NOAA to convert into pandas DataFrame
# format from http://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atlantic.pdf

import csv
import datetime
import pandas as pd


class HurricaneDataParserTestData:
    # Constants for validation when parsing
    STORM_HEADER_ROW_LENGTH = 3
    STORM_ENTRY_ROW_LENGTH = 20
    MISSING_PRESSURE_SYM = "-999"

    def __init__(self, filename="hurdat2_2017_2018.txt"):
        """
        Initializes the hurricane data API
        :param filename:
        """
        self.test_hurricanes = self._parse(filename)
        return

    def _parse(self, filename="hurdat2_2017_2018.txt"):
        """
        Parses hurricane data and stores in class member data frame.
        :param filename: Default filename.
        :return: Instantiated hurricanes data frame.
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