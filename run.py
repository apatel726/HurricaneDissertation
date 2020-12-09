from sklearn import model_selection
import numpy as np
import datetime
import argparse
import pprint

from hurricane_ai import data_utils
from hurricane_ai.ml.bd_lstm_td import BidrectionalLstmHurricaneModel

def log(message) :
    '''
    Creates a log system output in the message format below,
    [Timestamp] [HURAIM] : message
    Parameters
    ----------
    message str
        The message to log
    '''
    print(f'[{datetime.datetime.utcnow().isoformat()}Z] [HURAIM] {message}')

'''
Model Command Line Arguments
----------------------------
Create the model specified with the command line. e.g.
    >>> python run.py --universal
    >>> python run.py --singular
Accepts command line argument as either,
    universal
        Creates a universal model with wind intensity, lat, and long
    singular
        Creates singular models with 3 different models for wind, lat and long
If none are specified, we create a universal model
Training Command Line Arguments
-------------------------------
--load
    If there are models in the ml/models directory, we will use the files and weights in them according to the mode
        >>> python run.py --load                # loads the universal model weights
        >>> python run.py --singular --load     # loads the singular model weights
--epochs [int]
    The number of epochs to train the model
        >>> python run.py --singular --epochs 100
References
----------
https://docs.python.org/3/howto/argparse.html
'''
parser = argparse.ArgumentParser()
# flags for model
parser.add_argument("--singular", help = "The 'singular' version of the architecture will be used",
                    action = "store_true")
parser.add_argument("--universal", help = "The 'universal' version of the architecture will be used in ml/models",
                    action = "store_true")

# flags for the training
parser.add_argument("--load", help = "Loads existing model weights in the repository", action = "store_true")

# hyperparameters
parser.add_argument("--epochs", help = "Number of epochs to train the model", type = int, default = 1000)
parser.add_argument("--dropout", help = "The dropout hyperparameter", type = float, default = 0.01)
parser.add_argument("--loss", help = "The loss hyperparameter", default = 'mse')
parser.add_argument("--optimizer", help = "The optimizer hyperparameter", default = 'adam')

args = parser.parse_args()
log(str(args))


log('Creating scaled dataset')
scaled_train_val_data, feature_scaler = data_utils.build_scaled_ml_dataset(timesteps=5, load=args.load)

log('Creating our training and validation data structure')
X_train, X_val, y_train, y_val = model_selection.train_test_split(scaled_train_val_data['x'],
                                                                  scaled_train_val_data['y'],
                                                                  test_size=0.2,
                                                                  random_state=123)

def singular() :
    global y_train, y_val, X_train, X_val, args, feature_scaler
    # Wind intensity train/test features
    y_train_wind = data_utils.subset_features(y_train, 2)
    y_val_wind = data_utils.subset_features(y_val, 2)
    
    # Latitude/Longitude train/test features
    y_train_lat = data_utils.subset_features(y_train, 0)
    y_val_lat = data_utils.subset_features(y_val, 0)
    y_train_lon = data_utils.subset_features(y_train, 1)
    y_val_lon = data_utils.subset_features(y_val, 1)
    
    # Create and train bidirectional LSTM models for wind speed and track in isolation
    directory = f'singular_{datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M")}/'
    
    log('Create and train bidirectional LSTM wind model')
    bidir_lstm_model_wind = BidrectionalLstmHurricaneModel((X_train.shape[1], X_train.shape[2]), 'wind', feature_scaler,
                                                           dropout = args.dropout, loss = args.loss,
                                                           optimizer = args.optimizer, args = args)
    bidir_lstm_model_wind_hist = bidir_lstm_model_wind.train(X_train, y_train_wind, X_val, y_val_wind,
                                                             load_if_exists=args.load, epochs=args.epochs,
                                                             directory = directory)

    
    log('Create and train bidirectional LSTM lat model')
    bidir_lstm_model_lat = BidrectionalLstmHurricaneModel((X_train.shape[1], X_train.shape[2]), 'lat', feature_scaler,
                                                          dropout=args.dropout, loss=args.loss,
                                                          optimizer=args.optimizer, args=args)
    bidir_lstm_model_lat_hist = bidir_lstm_model_lat.train(X_train, y_train_lat, X_val, y_val_lat,
                                                           load_if_exists=args.load, epochs=args.epochs,
                                                             directory = directory)
    
    log('Create and train bidirectional LSTM lon model')
    bidir_lstm_model_lon = BidrectionalLstmHurricaneModel((X_train.shape[1], X_train.shape[2]), 'lon', feature_scaler,
                                                          dropout=args.dropout, loss=args.loss,
                                                          optimizer=args.optimizer, args=args)
    bidir_lstm_model_lon_hist = bidir_lstm_model_lon.train(X_train, y_train_lon, X_val, y_val_lon,
                                                           load_if_exists=args.load, epochs=args.epochs,
                                                             directory = directory)
    
    return {
        'wind' : (bidir_lstm_model_wind, bidir_lstm_model_wind_hist),
        'lat' : (bidir_lstm_model_lat, bidir_lstm_model_lat_hist),
        'lon' : (bidir_lstm_model_lon, bidir_lstm_model_lon_hist)
    }
            

def universal() :
    log('Create universal features')
    log('Train for wind intensity (index 0), lat (index 1), lon (index 2).')
    global y_train, y_val, X_train, X_val, args, feature_scaler
    y_train = np.array([[[features[2], features[0], features[1]] for features in y] for y in y_train], dtype = np.float64)
    y_val = np.array([[[features[2], features[0], features[1]] for features in y] for y in y_val], dtype = np.float64)

    log('Create and train bidirectional LSTM wind model')
    bidir_lstm_model_universal = BidrectionalLstmHurricaneModel((X_train.shape[1], X_train.shape[2]), 'universal',
                                                                feature_scaler, mode='universal', dropout=args.dropout,
                                                                loss=args.loss, optimizer=args.optimizer, args=args)
    bidir_lstm_model_universal_hist = bidir_lstm_model_universal.train(X_train, y_train, X_val, y_val,
                                                                       load_if_exists=args.load, epochs=args.epochs)

    return bidir_lstm_model_universal, bidir_lstm_model_universal_hist

if args.singular :
    model = singular()
elif args.universal :
    model = universal()
else :
    model = universal()
log('Completed this run!')