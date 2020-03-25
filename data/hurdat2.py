# Akash Patel
# Last Update: 3/25/30
# Read the text file from NOAA to convert into pandas DataFrame
# formate from http://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atlantic.pdf

import pandas as pd
import numpy as np
import datetime
import io
class hurdat2 :
    hurricanes = pd.DataFrame()

#initialize the hurdat 2 api

    def __init__(self, filename = "hurdat2.txt") :
        self.hurricanes = self.parse(filename)
        return
# parse the hurdat2 database


    def parse(self, filename = "hurdat2.txt", encoding ="utf-8") :
        db =[]
#parse line by line
        with open(filename) as raw :
            for line in raw :
                line = line.replace(' ', '').split(',')
#identify atlantic specific storms by "AL"
                if (line[0][:2] == 'AL') :
                    storm_id = line[0]
                    storm_name = line[1]
                    storm_entries = line[2]
#iterate and read through the track entries
                    for i in range(int(storm_entries)) :
                        entry = raw.readline().replace(' ', '').split(',')
#adding -999 for missing pressures                        
                        entry = [None if x == "-999" else x for x in entry]
#construct date and time entries based on columns
                        timestamp = datetime.datetime(int(entry[0][:4]),
                                                      int(entry[0][4:6]),
                                                      int(entry[0][6:8]),
                                                      int(entry[1][:2]),
                                                      int(entry[1][3:]))
                        db.append([storm_id, storm_name, timestamp] + entry[2:-1])
                else :
                            print("Error, unidentified storm ".join(str(line[0])))

                            return pd.DataFrame(db, columns = ['storm_id', 'storm_name', 'entry_time',
                                                               'entry_id', 'entry_status', 'lat', 'long', 'max_wind',
                                                               'min_pressure', '34kt_ne', '34kt_se', '34kt_sw', '34kt_nw',
                                                               '50kt_ne', '50kt_se', '50kt_sw', '64kt_se', '64kt_sw', '64kt_nw'])
                        
                    
    
    
