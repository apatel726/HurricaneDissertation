import tensorflow
import datetime
import pickle
import json
import os

# Base directory of project
PROJ_BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Raw input data file constants
HURRICANE_SOURCE_FILE = os.path.join(PROJ_BASE_DIR, 'data/source/hurdat2.txt')
ERROR_SOURCE_FILE = os.path.join(PROJ_BASE_DIR, 'data/source/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt')

# Processed data file constants
HURRICANE_PKL_FILE = os.path.join(PROJ_BASE_DIR, 'data/processed/hurricane_data.pkl')
HURRICANE_IDS_FILE = os.path.join(PROJ_BASE_DIR, 'data/processed/hurricane_ids.txt')
ERROR_PKL_FILE = os.path.join(PROJ_BASE_DIR, 'data/processed/error_data.pkl')
TRAIN_TEST_NPZ_FILE = os.path.join(PROJ_BASE_DIR, 'data/processed/train_test_data.npz')
SCALER_FILE = os.path.join(PROJ_BASE_DIR, 'scaler/feature_scaler.pkl')

# ML model constants
BD_LSTM_TD_MODEL = os.path.join(PROJ_BASE_DIR, 'models/bd_lstm_td_{}.h5')
BD_LSTM_TD_MODEL_HIST = os.path.join(PROJ_BASE_DIR, 'models/bd_lstm_td_{}_hist.csv')
LSTM_TD_MODEL = os.path.join(PROJ_BASE_DIR, 'models/lstm_td_{}.h5')
LSTM_TD_MODEL_HIST = os.path.join(PROJ_BASE_DIR, 'models/lstm_td_{}_hist.csv')


def is_source_modified(source_file, processed_file):
    """
    Determines whether the source file has been modified since the processed file was written.
    :param source_file: Source file.
    :param processed_file: Processed file.
    :return: Boolean indicator of whether the source has been modified.
    """
    return os.path.getmtime(source_file) > os.path.getmtime(processed_file)

def save(model, history, timestamp, prefix, args, scaler) :
    '''
    Uses HDF5 to save a directory for the models and a CSV for the history
    
    Parameters
    ----------
    model tf.keras.Model
        The model we will use to save
    history tf.keras.callbacks.History
        The history from the model
    prefix string
        The prefix of the filename. This can also be the file path
    timestamp datetime
        A datetime object for when the training started and is used to uniquely identify
        a model
    args dict
        A dictionary containing the arguments and hyperparameters
    scaler RobustScaler
        The scaler used to scale to and from the ground data
        
    References
    ----------
    https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format
    '''
    prefix = f'{prefix}{timestamp}/'
    
    # Create model name
    fname_model = f'{prefix}model_{timestamp}Z.h5'
    # Create history name
    fname_history = f'{prefix}model_history_{timestamp}Z.csv'
    
    # Save the model
    model.save(fname_model)
    
    # Save the history
    with open(fname_history, 'w+') as out_history:
        json.dump(str(history.history), out_history)

    # save the hyperparameters
    args['config'] = model.get_config()
    with open(f'{prefix}hyperparameters.json', 'w+') as hyperparameters :
        json.dump(args, hyperparameters)
        
    # save the scaler
    pickle.dump(scaler, open(f'{prefix}feature_scaler.pkl', 'wb' ))