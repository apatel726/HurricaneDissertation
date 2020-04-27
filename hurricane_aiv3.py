# Import various libraries throughout the software
import datetime
import math
import pickle as pkl
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential

# Import from hurdat2 class in container folder and models class from hurricane-models folder
from hurricane_ai.container.hurricane_data_container import HurricaneDataContainer
from hurricane_ai.container.error_model_container import ErrorModelContainer

# Initialize Dataframe for hurricanes and error database
dataset = HurricaneDataContainer("hurricane_ai/data/source/hurdat2.txt")  # Note that this container includes up to and including 2016
errors = ErrorModelContainer("hurricane_ai/data/source/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt")

# Show the first 5 records from Hurricane Katrina 2005 (AL122005)
dataset.hurricanes.query('storm_id == "AL122005"').head()

# Show the first 3 OFCL hurricane model errors for Hurricane Katrina 2005 on 28-08-2005/18:00:00
pprint(errors.error_models['OFCL'].storm['AL122005'][datetime.datetime(2005, 8, 28, 18, 0)], indent=8)


# Create hurricane class
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


# Storm ID Key for matching between datasets
storm_ids = dict()

# Parse in hurricanes
hurricanes = dict()
print("Transforming HURDAT2 into objects . . .")
for index, entry in dataset.hurricanes.iterrows():
    print("Transforming {}/{} entries from HURDAT2".format(index + 1, len(dataset.hurricanes)), end="\r")
    # New hurricane
    if entry['storm_id'] not in hurricanes:
        hurricanes[entry['storm_id']] = Hurricane(entry['storm_name'], entry['storm_id'])
        storm_ids[entry['storm_id']] = entry['storm_name']
    # Add entry to hurricane
    hurricanes[entry['storm_id']].add_entry(entry[2:])
print("\nDone!")

# Get all available model errors
models = errors.error_models.keys()
# Load model errors into hurricanes
for id in storm_ids:
    for model in models:
        # Skip if this hurricane does not have the model
        if id not in errors.error_models[model].storm:
            continue
        hurricanes[id].add_model(model, errors.error_models[model].storm[id])


def feature_extraction(timestep, previous):
    """
    PURPOSE: Calculate the features for a machine learning model within the context of hurricane-net
    METHOD: Use the predictors and the calculation methodology defined in Knaff 2013

    Timestep format:
    timestep = {
      'lat' : float,
      'long' : float,
      'max-wind' : float,
      'entry-time' : datetime
    }

    :param timestep: Current dictionary of features in the hurricane object format
    :param previous: Previous timestep dictionary of features in the hurricane object format
    :return: Dictionary of features
    """

    features = {
        'lat': timestep['lat'],
        'long': timestep['long'],
        'max_wind': timestep['max_wind'],
        'delta_wind': (timestep['max_wind'] - previous['max_wind']) /  # Calculated from track (12h)
                      ((timestep['entry_time'] - previous['entry_time']).total_seconds() / 43200),
        'min_pressure': timestep['min_pressure'],
        'zonal_speed': (timestep['lat'] - previous['lat']) /  # Calculated from track (per hour)
                       ((timestep['entry_time'] - previous['entry_time']).total_seconds() / 3600),
        'meridonal_speed': (timestep['long'] - previous['long']) /  # Calculated from track (per hour)
                           ((timestep['entry_time'] - previous['entry_time']).total_seconds() / 3600),
        'year': timestep['entry_time'].year,
        'month': timestep['entry_time'].month,
        'day': timestep['entry_time'].day,
        'hour': timestep['entry_time'].hour,
    }

    return features


def storm_x_y(storm, timesteps=1, lag=24):
    """
    PURPOSE: Create independent and dependent samples for a machine learning model based on the timesteps
    METHOD: Use the HURDAT2 database and a hurricane object as defined in hurricane-net for feature extraction

    :param storm: Hurricane object
    :param timesteps: Number of timesteps to calculate (default = 1)
    :param lag: Lag in hours for the dependent variables up to 5 days (default = 24)
    :return: Dictionary with independent (x) and dependent (y) values.
    """

    x = []
    # Create testing container structure with a dictionary
    times = [time * lag for time in
             range(1, (120 // lag) + 1)]  # Begin at lag hours with lag increments up to 120h inclusive
    y = dict([(time, []) for time in times])

    # Sort by entry time
    entries = [entry[1] for entry in sorted(storm.entries.items())]

    for index in range(len(entries)):
        if index < timesteps:  # Flag for insufficient initial time steps
            continue

        # If we're not including None values, check to see if there will be any
        if None in [storm.entries.get(entries[index]['entry_time'] +
                                      datetime.timedelta(hours=future)) for future in times]: break

        # Calculate time steps and their features for independent values
        sample = []
        for step in range(timesteps):
            # Training sample
            timestep = entries[index - step]
            previous = entries[index - step - 1]
            sample.append([timestep['entry_time']] + [[feature_extraction(timestep, previous)]])
        x.append(sample)  # Add our constructed sample

        # Calculate time steps and their features for dependent values
        for future in times:
            timestep = storm.entries.get(entries[index]['entry_time'] + datetime.timedelta(hours=future))
            previous = storm.entries.get(entries[index]['entry_time'] + datetime.timedelta(hours=future - lag))

            if timestep and previous:
                y[future].append(feature_extraction(timestep, previous))
            else:
                y[future].append(None)

    # Return output, if there is no output, return None.
    if len(x) is 0:
        return None
    else:
        return {'x': x, 'y': y}


def shape(hurricanes, timesteps, remove_missing=True):
    """
    PURPOSE: Shape our container for input into machine learning models
    METHOD: Use a numpy array to shape into (samples, timesteps, features)

    :param hurricanes: Dictionary of hurricane objects
    :param timesteps: Number of timesteps for the shape
    :param remove_missing: Boolean indicating whether the algorithm will disregard missing values
    :return: Numpy array of shape (samples, timesteps, 11) where 11 is the number of predictors in a hurricane object
    """

    x = []
    y = []
    lag = 24  # lag time in hours
    precision = np.float64  # defines the precision of our container type
    times = [time * lag for time in
             range(1, (120 // lag) + 1)]  # Begin at lag hours with lag increments up to 120h inclusive
    count = 0
    for hurricane in hurricanes.values():
        count += 1
        result = storm_x_y(hurricane, timesteps, lag)
        if result is None:
            continue
        # Extract only the values from the strom features using our specified precision
        hurricane_x = np.array(
            [[list(sample[1][0].values()) for sample in x] for x in result['x']],
            dtype=precision)
        hurricane_y = np.array(
            [[list(result['y'][time][index].values()) for time in times] for index in range(len(result['y'][lag]))],
            dtype=precision)
        # Disregard if algorithm requires no missing values
        if remove_missing:
            if (len(np.where(np.isnan(hurricane_x))[0]) > 0) or (len(np.where(np.isnan(hurricane_y))[0]) > 0):
                continue
        # Add to our results
        x.extend(hurricane_x)
        y.extend(hurricane_y)
        print("Feature engineered {}/{} hurricanes for {} timestep(s)".format(count, len(hurricanes), timesteps),
              end="\r")
    print("\nDone feature engineering hurricanes.")

    return {'x': np.array(x), 'y': np.array(y)}


def fit_feature_scaler(processed_data, hurricanes):
    """
    PURPOSE: Scale our container using the RobustScaler method from the sklearn library
    METHOD: Generate container using 1 timesteps and then remove the NaN or None types to use the scaler methods

    :param processed_data: Dictionary of x and y values of container produced by shape() function with no missing values
    :param hurricanes: Dictionary of hurricane objects
    :return: 1) Scaled processed_data using RobustScaler
             2) RobustScaler object fit with appropriate container
    """

    print("Scaling Data . . . (1 timestep for unqiue container)")

    # Create scaler
    unique_data = shape(hurricanes, timesteps=1)
    x = np.reshape(unique_data['x'], (unique_data['x'].shape[0], -1))
    x = np.delete(x, np.where(np.isnan(x))[0], 0)
    feature_scaler = RobustScaler()
    feature_scaler.fit(x)

    # Scale our container
    for index in range(len(processed_data['x'])):
        # Scale our x
        processed_data['x'][index] = feature_scaler.transform(processed_data['x'][index])
        # Scale our y
        processed_data['y'][index] = feature_scaler.transform(processed_data['y'][index])

    print("Done scaling.")

    return processed_data, feature_scaler


# Finalize and scale procesed container into a dictionary
preprocessed_data = shape(hurricanes, timesteps=5)
processed_data, scaler = fit_feature_scaler(preprocessed_data, hurricanes)

# Save fitted scaler
with open(r'scalers/hurricane_scaler.pkl', 'wb') as out_file:
    pkl.dump(scaler, out_file)

# Create our cross validation container structure
X_train, X_test, y_train, y_test = model_selection.train_test_split(processed_data['x'], processed_data['y'],
                                                                    test_size=0.2)

# Train for wind intensity
y_train_wind = np.array([[[features[2]] for features in y] for y in y_train], dtype=np.float64)
y_test_wind = np.array([[[features[2]] for features in y] for y in y_test], dtype=np.float64)

# Train for latitude and longitude location
y_train_lat = np.array([[[features[0]] for features in y] for y in y_train], dtype=np.float64)
y_test_lat = np.array([[[features[0]] for features in y] for y in y_test], dtype=np.float64)
y_train_long = np.array([[[features[1]] for features in y] for y in y_train], dtype=np.float64)
y_test_long = np.array([[[features[1]] for features in y] for y in y_test], dtype=np.float64)


def bd_lstm_td(X_train, y_train, X_test, y_test, n_epochs=500):
    """
    Instantiates and trains a bidirectional LSTM model.

    :param X_train: Training set observations
    :param y_train: Training set labels
    :param X_test: Test set observations
    :param y_test: Test set labels
    :param n_epochs: Number of epochs to train
    :return: The trained model
    """

    model = Sequential()
    model.add(Bidirectional(LSTM(units=512, return_sequences=True, dropout=0.05),
                            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=256, return_sequences=True, dropout=0.05))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adadelta')

    print(model.summary())

    history = model.fit(X_train, y_train, batch_size=len(X_train), epochs=n_epochs,
                        validation_data=(X_test, y_test))

    return model, history


def lstm_td(X_train, X_test, y_train, y_test):
    """
    Instantiates and trains an LSTM model.

    :param X_train: Training set observations
    :param y_train: Training set labels
    :param X_test: Test set observations
    :param y_test: Test set labels
    :param n_epochs: Number of epochs to train
    :return: The trained model
    """

    model = Sequential()
    model.add(LSTM(units=1024, input_shape=(5, 8), return_sequences=True))
    model.add(TimeDistributed(Dense(8)))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())

    model.fit(X_train, y_train, batch_size=len(X_train), epochs=300)

    return model


model_wind, model_wind_history = bd_lstm_td(X_train, y_train_wind, X_test, y_test_wind, n_epochs=500)
model_lat, model_lat_history = bd_lstm_td(X_train, y_train_lat, X_test, y_test_lat, n_epochs=1000)
model_long, model_long_history = bd_lstm_td(X_train, y_train_long, X_test, y_test_long, n_epochs=1000)

# Save models
model_wind.save('models/huraim_wind.h5')
model_lat.save('models/huraim_lat.h5')
model_long.save('models/huraim_long.h5')

def ai_errors(predictions, observations, history=None):
    """
    PURPOSE: Provide descriptive statistics on the predicted output versus the observed measurments
    METHOD:  Take the errors of the predictions and answers and then calculate standard descriptive statistics

    :param predictions: 2D array of predictions of observed output
    :param observations: 2D array measurements of observed output
    :param history: Keras history model for displaying model loss, default is None if not available
    :return: Data frame of model prediction errors
    """

    errors = []
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            # Calculate errors
            error = predictions[i][j] - observations[i][j]
            errors.append(error)

    # Display history and erros
    plt.figure(1)
    plt.hist(errors, bins=50)
    plt.title('error histogram')
    plt.xlabel('error')
    plt.ylabel('frequency')
    plt.grid(True)

    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return pd.DataFrame(errors)


# Predict values
wind_predictions = model_wind.predict(X_test)
lat_predictions = model_lat.predict(X_test)
long_predictions = model_long.predict(X_test)

# Scale back our predictions
# Wind
wind_predictions_scaled = [scaler.inverse_transform([[0, 0, winds[0], 0, 0, 0, 0, 0, 0, 0, 0] for winds in prediction])
                           for prediction in wind_predictions]
y_wind_test_scaled = [scaler.inverse_transform([[0, 0, winds[0], 0, 0, 0, 0, 0, 0, 0, 0] for winds in observation])
                      for observation in y_test_wind]
# Latitude
lat_predictions_scaled = [scaler.inverse_transform([[lat[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for lat in prediction])
                          for prediction in lat_predictions]
y_lat_test_scaled = [scaler.inverse_transform([[lat[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for lat in observation])
                     for observation in y_test_lat]
# Longitude
long_predictions_scaled = [scaler.inverse_transform([[0, long[0], 0, 0, 0, 0, 0, 0, 0, 0, 0] for long in prediction])
                           for prediction in long_predictions]
y_long_test_scaled = [scaler.inverse_transform([[0, long[0], 0, 0, 0, 0, 0, 0, 0, 0, 0] for long in observation])
                      for observation in y_test_long]

# Record wind predictions and observations
print("Wind")
wind_predictions = [[pred[2] for pred in hurricanes_pred] for hurricanes_pred in wind_predictions_scaled]
wind_observations = [[obsrv[2] for obsrv in hurricanes_obsrv] for hurricanes_obsrv in y_wind_test_scaled]

# Present Errors
ai_errors(wind_predictions, wind_observations, model_wind_history).describe()

print("Lat")
lat_predictions = [[pred[0] for pred in hurricanes_pred] for hurricanes_pred in lat_predictions_scaled]
lat_observations = [[obsrv[0] for obsrv in hurricanes_obsrv] for hurricanes_obsrv in y_lat_test_scaled]
ai_errors(lat_predictions, lat_observations, model_lat_history).describe()

print("Long")
long_predictions = [[pred[1] for pred in hurricanes_pred] for hurricanes_pred in long_predictions_scaled]
long_observations = [[obsrv[1] for obsrv in hurricanes_obsrv] for hurricanes_obsrv in y_long_test_scaled]
ai_errors(long_predictions, long_observations, model_long_history).describe()

test_data = HurricaneDataContainer('hurricane_ai/data/source/hurdat2-1851-2017-050118.txt')

# Parse in hurricanes
hurricanes_2017 = dict()
print("Transforming 2017 HURDAT2 into objects . . .")
for index, entry in test_data.hurricanes.iterrows():
    print("Transforming {}/{} entries from HURDAT2".format(index + 1, len(dataset.hurricanes)), end="\r")
    # Filter to capture 2017 container
    if entry['storm_id'][-4:] != '2017':
        continue
    if entry['storm_id'] not in hurricanes_2017:
        hurricanes_2017[entry['storm_id']] = Hurricane(entry['storm_name'], entry['storm_id'])
        storm_ids[entry['storm_id']] = entry['storm_name']
    # Add entry to hurricane
    hurricanes_2017[entry['storm_id']].add_entry(entry[2:])
print("\nDone!")

# Filter storms that have more than 6 entries. We need at least 6 to calculate 5 speed vectors
storms_filter = [storm for storm in hurricanes_2017.values() if len(storm.entries) > 6]

# Begin creating hurricane forecast and track predictions
tracks = {
    'storms': [],  # Reference storm
    'inputs': [],  # The inputs for the ai
    'valid_times': [],  # The valid time to compare to the error database
}
for index, storm in enumerate(storms_filter):
    # Create inputs to ai. ai requires scaled container as input
    entries = [entry[1] for entry in sorted(storm.entries.items())]  # Extracts container from container structure

    # Scale the entries
    for start_index in range(1, len(entries) - 5):  # Go through each entry
        # Build feature extraction
        extracted_features = []
        valid_time = None  # Going to be set to the last element in the series
        for pivot in range(start_index, start_index + 5):
            extracted_features.append(np.array(list(feature_extraction(entries[pivot], entries[pivot - 1]).values())))
            if pivot is start_index + 4:  # We're on the last element
                valid_time = entries[pivot]['entry_time']

        # If there's an incomplete value we can't process, skip it
        if any(None in entry for entry in extracted_features):
            continue

        # Scale extracted features
        scaled_entries = scaler.transform(extracted_features)

        # Add to our results
        tracks['storms'].append(storm)
        tracks['inputs'].append(scaled_entries.tolist())
        tracks['valid_times'].append(valid_time)

    print("\rDone with track processing {}/{} storms".format(index + 1, len(storms_filter)), end='')

tracks['inputs'] = np.array(tracks['inputs'])
tracks['wind_predictions_raw'] = model_wind.predict(tracks['inputs'])
tracks['lat_predictions_raw'] = model_lat.predict(tracks['inputs'])
tracks['long_predictions_raw'] = model_long.predict(tracks['inputs'])


def distance(origin, destination):
    """
    Returns the distance between two coordinates in nautical miles.

    :param origin: The origin point
    :param destination: The destination point
    :return: The distance between the two points in nautical miles
    """

    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d * 0.539957  # km to nautical miles


# Scale back and store our wind predictions and our lat, long predictions
tracks['wind_predictions'] = []
tracks['lat_predictions'] = []
tracks['long_predictions'] = []
intensity_errors = {
    '24': [],
    '48': [],
    '72': [],
    '96': [],
    '120': []
}
track_errors = {
    '24': [],
    '48': [],
    '72': [],
    '96': [],
    '120': []
}
for index, prediction in enumerate(tracks['wind_predictions_raw']):
    # Use our standard scaler to scale the raw predictions back
    winds_scaled = [scaler.inverse_transform(
        [[0, 0, winds[0], 0, 0, 0, 0, 0, 0, 0, 0] for winds in prediction])]  # Index 2 is winds
    lat_scaled = [scaler.inverse_transform(
        [[lats[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for lats in tracks['lat_predictions_raw'][index]])]  # Index 0 is lat
    long_scaled = [scaler.inverse_transform([[0, longs[0], 0, 0, 0, 0, 0, 0, 0, 0, 0] for longs in
                                             tracks['long_predictions_raw'][index]])]  # Index 1 is long

    # Extract the wind prediction from container structure and store into new container structure
    for i in range(len(winds_scaled)):
        # The new container structure is a tuple of (wind, storm_id, valid_time, forecast_time)
        wind_predictions = []
        lat_predictions = []
        long_predictions = []
        for step, pred in enumerate(winds_scaled[i]):
            wind = pred[2]
            lat = lat_scaled[i][step][0]
            long = long_scaled[i][step][1]

            storm_id = tracks['storms'][index].id
            valid_time = tracks['valid_times'][index]
            forecast_time = valid_time + datetime.timedelta(days=step + 1)

            # See if we can find the error
            if forecast_time in hurricanes_2017[storm_id].entries:
                wind_truth = hurricanes_2017[storm_id].entries[forecast_time]['max_wind']
                lat_truth = hurricanes_2017[storm_id].entries[forecast_time]['lat']
                long_truth = hurricanes_2017[storm_id].entries[forecast_time]['long']
                intensity_error = abs(wind_truth - wind)
                track_error = distance((lat_truth, long_truth), (lat, long))

                wind_predictions.append({
                    'ai-wind': wind,
                    'truth': wind_truth,
                    'storm_id': storm_id,
                    'valid_time': valid_time,
                    'forecast_time': forecast_time
                })
                lat_predictions.append({
                    'ai-lat': lat,
                    'truth': lat_truth,
                    'storm_id': storm_id,
                    'valid_time': valid_time,
                    'forecast_time': forecast_time
                })
                long_predictions.append({
                    'ai-long': long,
                    'truth': long_truth,
                    'storm_id': storm_id,
                    'valid_time': valid_time,
                    'forecast_time': forecast_time
                })
                if step is 0:
                    intensity_errors['24'].append(intensity_error)
                    track_errors['24'].append(track_error)
                if step is 1:
                    intensity_errors['48'].append(intensity_error)
                    track_errors['48'].append(track_error)
                if step is 2:
                    intensity_errors['72'].append(intensity_error)
                    track_errors['72'].append(track_error)
                if step is 3:
                    intensity_errors['96'].append(intensity_error)
                    track_errors['96'].append(track_error)
                if step is 4:
                    intensity_errors['120'].append(intensity_error)
                    track_errors['120'].append(track_error)

        tracks['wind_predictions'].append(wind_predictions)
        tracks['lat_predictions'].append(lat_predictions)
        tracks['long_predictions'].append(long_predictions)

pd.DataFrame(intensity_errors['24']).describe()
pd.DataFrame(intensity_errors['48']).describe()
pd.DataFrame(intensity_errors['72']).describe()
pd.DataFrame(intensity_errors['96']).describe()
pd.DataFrame(intensity_errors['120']).describe()

pd.DataFrame(track_errors['24']).describe()
pd.DataFrame(track_errors['48']).describe()
pd.DataFrame(track_errors['72']).describe()
pd.DataFrame(track_errors['96']).describe()
pd.DataFrame(track_errors['120']).describe()

errordb = errors.error_model_container.ErrorModelContainer(
    "errors/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt")
ai_wind_errors = []
ai_track_errors = []
bcd5_wind_errors = []
bcd5_track_errors = []
for index, prediction in enumerate(tracks['wind_predictions']):
    # Find the time stamp for the storm ID in the error database
    if prediction == []:
        continue
    valid_time = prediction[0]['valid_time']
    storm_id = prediction[0]['storm_id']
    # Check to see if we have error for this storm and at the valid time
    if storm_id in errordb.error_models['BCD5'].storm and valid_time in errordb.error_models['BCD5'].storm[storm_id]:
        print("Found {} at {}".format(storm_id, valid_time))
        # If we find it, compare
        for i, forecast in enumerate(prediction):
            # See if we can find another prediction like that in the error database
            if errordb.error_models['BCD5'].storm[storm_id][valid_time]['intensity_forecast'][
                forecast['forecast_time'].to_pydatetime()]:
                print("\tIntensity Truth: {}, AI forecast: {}, BCD5 forecast: {}".format(forecast['truth'],
                                                                                         forecast['ai-wind'],
                                                                                         errordb.error_models[
                                                                                             'BCD5'].storm[
                                                                                             storm_id][valid_time][
                                                                                             'track_forecast'][forecast[
                                                                                             'forecast_time'].to_pydatetime()]))
                print("\tTrajectory Truth: {}, {}; AI forecast: {}, {} ; AI error: {} BCD5 error: {}".format(
                    tracks['lat_predictions'][index][i]['truth'],
                    tracks['long_predictions'][index][i]['truth'],
                    tracks['lat_predictions'][index][i]['ai-lat'],
                    tracks['long_predictions'][index][i]['ai-long'],
                    distance(
                        (tracks['lat_predictions'][index][i]['truth'], tracks['long_predictions'][index][i]['truth']), (
                            tracks['lat_predictions'][index][i]['ai-lat'],
                            tracks['long_predictions'][index][i]['ai-long'])),
                    errordb.error_models['BCD5'].storm[storm_id][valid_time]['intensity_forecast'][
                        forecast['forecast_time'].to_pydatetime()]
                ))
                ai_wind_errors.append(abs(forecast['truth'] - forecast['ai-wind']))
                ai_track_errors.append(abs(distance(
                    (tracks['lat_predictions'][index][i]['truth'], tracks['long_predictions'][index][i]['truth']),
                    (tracks['lat_predictions'][index][i]['ai-lat'], tracks['long_predictions'][index][i]['ai-long']))))
                bcd5_wind_errors.append(abs(errordb.error_models['BCD5'].storm[storm_id][valid_time]['track_forecast'][
                                                forecast['forecast_time'].to_pydatetime()]))
                bcd5_track_errors.append(
                    abs(errordb.error_models['BCD5'].storm[storm_id][valid_time]['intensity_forecast'][
                            forecast['forecast_time'].to_pydatetime()]))

pd.DataFrame(ai_wind_errors).describe()
pd.DataFrame(bcd5_wind_errors).describe()
pd.DataFrame(ai_track_errors).describe()
pd.DataFrame(bcd5_track_errors).describe()
tracks['inputs'][:1]

[output[2] for output in
 scaler.inverse_transform(
     [[0, 0, winds[0], 0, 0, 0, 0, 0, 0, 0, 0] for winds in model_wind.predict(tracks['inputs'][:1])[0]]
 )]


def hurricane_ai(input):
    '''
    input = {
      -120 : timestep,
      -96 : timestep,
      -72 : timestep,
      -48 : timestep,
      -24 : timestep,
      0 : timestep
    }
    output = {
      24 : prediction,
      48 : prediction,
      72 : prediction,
      96 : prediction,
      120 : prediction
    }
    timestep = {
        'lat' : float,
        'long' : float,
        'max-wind' : float,
        'min_pressure' : float,
        'entry-time' : datetime
    }
    prediction = {
      'lat' : float,
      'long' : float,
      'max-winds' : float
    }
    '''
    # Take entries and transform them into our container model
    extract = []
    temp = None
    for index, value in enumerate([-120, -96, -72, -48, -24, 0]):
        if not index:
            temp = input[value]
            continue
        else:
            extract.append(list(feature_extraction(input[value], temp).values()))
            temp = input[value]

    state = np.expand_dims(fit_feature_scaler.transform(extract), axis=0)
    print('extract: {}, state: {}'.format(extract, state))
    # Finally, use our hurricane ai to predict storm state
    lat = [output[0] for output in fit_feature_scaler.inverse_transform(
        [[lat[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for lat in model_lat.predict(state)[0]])]
    long = [output[1] for output in fit_feature_scaler.inverse_transform(
        [[0, long[0], 0, 0, 0, 0, 0, 0, 0, 0, 0] for long in model_long.predict(state)[0]])]
    wind = [output[2] for output in fit_feature_scaler.inverse_transform(
        [[0, 0, wind[0], 0, 0, 0, 0, 0, 0, 0, 0] for wind in model_wind.predict(state)[0]])]

    output = dict()
    for index, value in enumerate([24, 48, 72, 96, 120]):
        output[value] = {
            'lat': lat[index],
            'long': long[index],
            'max_wind': wind[index]
        }

    return output


input = {
    0: {
        'entry_time': parse('Fri Aug 30 2019 1100 PM'),
        'lat': 25.5,
        'long': 71.4,
        'max_wind': 140 / 1.51,  # mph to knots
        'min_pressure': 948.0
    },
    -24: {
        'entry_time': parse('Thu Aug 29 2019 1100 PM'),
        'lat': 23.3,
        'long': 68.4,
        'max_wind': 105 / 1.51,  # mph to knots
        'min_pressure': 977.0
    },
    -48: {
        'entry_time': parse('Wed Aug 28 2019 1100 PM'),
        'lat': 19.7,
        'long': 66.0,
        'max_wind': 85 / 1.51,  # mph to knots
        'min_pressure': 986.0
    },
    -72: {
        'entry_time': parse('Tue Aug 27 2019 1100 PM'),
        'lat': 16.0,
        'long': 63.0,
        'max_wind': 50 / 1.51,  # mph to knots
        'min_pressure': 1006.0
    },
    -96: {
        'entry_time': parse('Mon Aug 26 2019 1100 PM'),
        'lat': 13.2,
        'long': 59.7,
        'max_wind': 50 / 1.51,  # mph to knots
        'min_pressure': 1003.0
    },
    -120: {
        'entry_time': parse('Sun Aug 25 2019 1100 PM'),
        'lat': 11.7,
        'long': 55.3,
        'max_wind': 50 / 1.51,  # mph to knots
        'min_pressure': 1003.0
    }
}

hurricane_ai(input)
