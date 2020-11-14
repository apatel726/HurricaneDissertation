# -*- coding: utf-8 -*-
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
import requests
from datetime import datetime
import dateutil.parser
from pytz import timezone
import zipfile
import io
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