import json
import logging
from os import path
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed

from hurricane_ai import BD_LSTM_TD_MODEL, BD_LSTM_TD_MODEL_HIST


class BidrectionalLstmHurricaneModel:
    """
    Class encapsulating a single-output bi-directional LSTM hurricane model.
    """

    def __init__(self, shape, predicted_var: str, loss='mse', optimizer='adadelta', validation_split=0.2, mode='singular', dropout=0.05):
        """
        Set default training parameters and instantiate the model architecture.
        :param shape: The input shape.
        :param predicted_var: The name of the variable being predicted (forecast) by the model.
        :param loss: The loss function.
        :param optimizer: The optimizer.
        :param validation_split: The percentage of the training dataset to use for validation.
        """
        self.input_shape = shape
        self.predicted_var = predicted_var
        self.loss = loss
        self.optimizer = optimizer
        self.validation_split = validation_split
        self.mode = mode
        self.dropout = dropout
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
        model.compile(loss=self.loss, optimizer=self.optimizer)

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

        # Train model
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_split=self.validation_split, verbose=verbose)

        # Serialize model weights
        os.makedirs(os.path.dirname(weight_file), exist_ok=True)
        self.model.save_weights(weight_file)

        # Serialize history to CSV
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, 'w') as out_file:
            json.dump(history.history, out_file)

        return history.history
