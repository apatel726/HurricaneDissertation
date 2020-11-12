import json
import logging
import datetime
from os import path
import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import load_model

from typing import Union

from hurricane_ai import BD_LSTM_TD_MODEL, BD_LSTM_TD_MODEL_HIST, save

class BidrectionalLstmHurricaneModel:

    FEATURES = [
        "lat",
        "lon",
        "max_wind",
        "delta_wind",
        "min_pressure",
        "zonal_speed",
        "meridonal_speed",
        "year",
        "month",
        "day",
        "hour"
    ]

    """
    Class encapsulating a single-output bi-directional LSTM hurricane model.
    """

    def __init__(self, shape, predicted_var: str, scaler: Union[RobustScaler, str], loss='mse', optimizer='adadelta',
                 validation_split=0.2, mode='singular', dropout=0.05, args={}, model_path=None):
        """
        Set default training parameters and instantiate the model architecture.
        :param shape: The input shape.
        :param predicted_var: The name of the variable being predicted (forecast) by the model.
        :param scaler: The scaler object or path to load serialized scaler file.
        :param loss: The loss function.
        :param optimizer: The optimizer.
        :param validation_split: The percentage of the training dataset to use for validation.
        :param mode: universal or singular architecture
        :param args: the command line arguments containing hyperparameters
        :param model_path: The path to the serialized model file
        """
        self.input_shape = shape
        self.predicted_var = predicted_var
        self.loss = loss
        self.optimizer = optimizer
        self.validation_split = validation_split
        self.mode = mode
        self.dropout = dropout
        self.args = args

        # Load the feature scaler
        if isinstance(scaler, RobustScaler):
            self.scaler = scaler
        elif isinstance(scaler, str) and os.path.exists(scaler):
            with open(scaler, 'rb') as in_file:
                self.scaler = pkl.load(in_file)
        else:
            raise ValueError(
                "scaler argument must be either a RobustScaler object or a valid path to a serialized scaler")

        # Load model if one is specified, otherwise construct a new one for training
        if model_path:
            self.model = load_model(model_path)
        else:
            self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """
        Build and compile the model architecture.
        :return: The compiled model architecture.
        """

        model = Sequential()
        model.add(Bidirectional(LSTM(units=512, return_sequences=True, dropout=self.dropout), input_shape=self.input_shape))
        model.add(LSTM(units=256, return_sequences=True, dropout=0.05))
        if self.mode == 'singular' :
            model.add(TimeDistributed(Dense(1)))
        elif self.mode == 'universal' :
            model.add(TimeDistributed(Dense(3)))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics = [tf.keras.metrics.MeanAbsoluteError(),
                                                                          tf.keras.metrics.MeanAbsolutePercentageError()])

        return model

    def train(self, X_train, y_train, batch_size=5000, epochs=50, load_if_exists=True, verbose=True) -> dict:
        """
        Train the model using the given dataset and parameters.
        :param X_train: The training dataset observations.
        :param y_train: The training dataset labels.
        :param batch_size: The number of observations in a mini-batch.
        :param epochs: The number of epochs.
        :param load_if_exists: Indicates whether model should be loaded from disk if it exists.
        :param verbose: Indicates whether to use verbose output during training.
        :return: The training history.
        """

        # Derive weight and history filenames based on predicted variable
        weight_file = BD_LSTM_TD_MODEL.format(self.predicted_var)
        history_file = BD_LSTM_TD_MODEL_HIST.format(self.predicted_var)

        if load_if_exists and path.exists(weight_file):
            # Load the serialized model weights
            self.model.load_weights(weight_file)

            # Load the training history
            with open(history_file, 'r') as in_file:
                history = json.load(in_file)

            return history

        # create model directory
        timestamp = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
        prefix = 'hurricane_ai/models/'
        logs = tf.keras.callbacks.TensorBoard(log_dir = f'{prefix}{timestamp}/', histogram_freq = 1,
                                              profile_batch = 0)
        
        # Train model
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_split=self.validation_split, verbose=verbose, 
                                 callbacks = [logs])

        # Save model and history
        save(self.model, history, timestamp, prefix, vars(self.args), self.scaler)
        

        return history.history

    def predict(self, observation_df: pd.DataFrame, timesteps: int) -> float:
        """
        Runs inference on the given observation data frame.

        :param observation_df: Ground truth hurricane measurements collected to date.
        :param timesteps: Number of timesteps over which to run inference.
        :return: Predicted value (e.g. lat, lon, wind)
        """

        # Subset to relevant features
        feature_df = observation_df[self.FEATURES]

        # Truncate to last
        truncated_df = feature_df.tail(timesteps)

        # Normalize data
        normalized_features = self.scaler.transform(truncated_df.values)

        # Add batch dimension (just 1 for single pass inference)
        feature_values = np.expand_dims(normalized_features, 0)
        print(f'feature values shape: {feature_values.shape}')
        # Run inference and extract predictions
        predictions = np.squeeze(self.model.predict(feature_values))

        # Extract and return single prediction
        return predictions
