from sklearn import model_selection

from hurricane_ai import data_utils
from hurricane_ai.ml.bd_lstm_td import BidrectionalLstmHurricaneModel

scaled_train_test_data, feature_scaler = data_utils.build_scaled_ml_dataset(timesteps=5)

# Create our cross validation data structure
X_train, X_test, y_train, y_test = model_selection.train_test_split(scaled_train_test_data['x'],
                                                                    scaled_train_test_data['y'], test_size=0.2)

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
bidir_lstm_model_lon_hist = bidir_lstm_model_lon.train(X_train, y_train_lon)
