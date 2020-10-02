from sklearn import model_selection
import numpy as np
import argparse # for command line arguments

from hurricane_ai import data_utils
from hurricane_ai.ml.bd_lstm_td import BidrectionalLstmHurricaneModel

scaled_train_test_data, feature_scaler = data_utils.build_scaled_ml_dataset(timesteps=5)

# Create our cross validation data structure
X_train, X_test, y_train, y_test = model_selection.train_test_split(scaled_train_test_data['x'],
                                                                    scaled_train_test_data['y'], test_size=0.2)

'''
Create the model specified with the command line. e.g.
    >>> python run.py universal
    >>> python run.py singular
Accepts 1 command line as either,
    universal
        Creates a universal model with wind intensity, lat, and long
    singular
        Creates singular models with 3 different models for wind, lat and long
If none are specified, we create a universal model

Other command line arguments include,

References
----------
https://docs.python.org/2/howto/argparse.html

'''
parser = argparse.ArgumentParser()
# flags for model
parser.add_argument("--singular", help = "The 'singular' version of the architecture will be used",
                    action = "store_true")
parser.add_argument("--universal", help = "The 'universal' version of the architecture will be used",
                    action = "store_true")

# flags for the training
parser.add_argument("--load", help = "Loads existing model weights in the repository", action = "store_true")

args = parser.parse_args()

def singular() :
    global y_train, y_test, X_train, X_test, args
    # Wind intensity train/test features
    y_train_wind = data_utils.subset_features(y_train, 2)
    y_test_wind = data_utils.subset_features(y_test, 2)
    
    # Latitude/Longitude train/test features
    y_train_lat = data_utils.subset_features(y_train, 0)
    y_test_lat = data_utils.subset_features(y_test, 0)
    y_train_lon = data_utils.subset_features(y_train, 1)
    y_test_lon = data_utils.subset_features(y_test, 1)
    
    # Create and train bidirectional LSTM models for wind speed and track in isolation
    
    # Create and train bidirectional LSTM wind model
    bidir_lstm_model_wind = BidrectionalLstmHurricaneModel((X_train.shape[1], X_train.shape[2]), 'wind')
    bidir_lstm_model_wind_hist = bidir_lstm_model_wind.train(X_train, y_train_wind)
    
    # Create and train bidirectional LSTM track model
    bidir_lstm_model_lat = BidrectionalLstmHurricaneModel((X_train.shape[1], X_train.shape[2]), 'lat')
    bidir_lstm_model_lat_hist = bidir_lstm_model_lat.train(X_train, y_train_lat)
    bidir_lstm_model_lon = BidrectionalLstmHurricaneModel((X_train.shape[1], X_train.shape[2]), 'lon')
    bidir_lstm_model_lon_hist = bidir_lstm_model_lon.train(X_train, y_train_lon, load_if_exists = args.load)
    
    return {
        'wind' : (bidir_lstm_model_wind, bidir_lstm_model_wind_hist),
        'lat' : (bidir_lstm_model_lat, bidir_lstm_model_lat_hist),
        'long' : (bidir_lstm_model_lon, bidir_lstm_model_lon_hist)
        }

def universal() :
    # Create universal features
    # Train for wind intensity (index 0), lat (index 1), long (index 2).
    global y_train, y_test, X_train, X_test, args
    y_train = np.array([[[features[2], features[0], features[1]] for features in y] for y in y_train], dtype = np.float64)
    y_test = np.array([[[features[2], features[0], features[1]] for features in y] for y in y_test], dtype = np.float64)

    # Create and train bidirectional LSTM wind model
    bidir_lstm_model_universal = BidrectionalLstmHurricaneModel((X_train.shape[1], X_train.shape[2]), 'universal', mode = 'universal')
    bidir_lstm_model_universal_hist = bidir_lstm_model_universal.train(X_train, y_train, load_if_exists = args.load)
    
    return bidir_lstm_model_universal, bidir_lstm_model_universal_hist

if args.singular :
    model = singular()
elif args.universal :
    model = universal()
else :
    model = universal()