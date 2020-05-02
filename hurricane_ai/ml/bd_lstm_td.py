from os import path
import json
import logging
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed

from hurricane_ai import BD_LSTM_TD_MODEL, BD_LSTM_TD_MODEL_HIST


class BidrectionalLstmHurricaneModel:
    """
    Class encapsulating a single-output bi-directional LSTM hurricane model.
    """

    def __init__(self, shape, loss='mse', optimizer='adadelta', validation_split=0.2):
        """
        Set default training parameters and instantiate the model architecture.
        :param shape: The input shape.
        :param loss: The loss function.
        :param optimizer: The optimizer.
        :param validation_split: The percentage of the training dataset to use for validation.
        """
        self.input_shape = shape
        self.loss = loss
        self.optimizer = optimizer
        self.validation_split = validation_split
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """
        Build and compile the model architecture.
        :return: The compiled model architecture.
        """

        model = Sequential()
        model.add(Bidirectional(LSTM(units=512, return_sequences=True, dropout=0.05), input_shape=self.input_shape))
        model.add(LSTM(units=256, return_sequences=True, dropout=0.05))
        model.add(TimeDistributed(Dense(1)))
        model.compile(loss=self.loss, optimizer=self.optimizer)

        logging.debug('Compiled bidirectional LSTM model')

        return model

    def train(self, X_train, y_train, batch_size=5000, epochs=1000, load_if_exists=True, verbose=False) -> dict:
        """
        Train the model using the given dataset and parameters.
        :param X_train: The training dataset observations.
        :param y_train: The training dataset labels.
        :param batch_size: The number of observations in a mini-batch.
        :param epochs: The number of epochs.
        :param load_if_exists: Indicates whether model should be loaded from disk if it exists.
        :return: The training history.
        """

        if load_if_exists and path.exists(BD_LSTM_TD_MODEL):
            # Load the serialized model weights
            self.model.load_weights(BD_LSTM_TD_MODEL)

            # Load the training history
            with open(BD_LSTM_TD_MODEL_HIST, 'r') as in_file:
                history = json.load(in_file)

            return history

        # Train model
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_split=self.validation_split, verbose=verbose)

        # Serialize history to CSV
        with open(BD_LSTM_TD_MODEL_HIST, 'w') as out_file:
            json.dump(history.history, out_file)

        return history.history
