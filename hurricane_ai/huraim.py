import argparse
import sys


class HurricaneModeler:

    def __init__(self, **kwargs):
        self.input_data = kwargs['input_data'] if 'input_data' in kwargs.keys() else 'data/hurdat2.txt'
        self.error_data = kwargs[
            'error_data'] if 'error_data' in kwargs.keys() else 'errors/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt'
        self.retrain = kwargs['retrain'] if 'retrain' in kwargs.keys() else False


def parse_args():

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_data', default='data/hurdat2.txt',
                        type=str, help='Historical Hurricane Track data')
    parser.add_argument('--error_data', default='errors/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt',
                        type=str, help='Hurricane track errors')
    parser.add_argument('--retrain', action='store_true',
                        dest='retrain', help='Retrains the model(s) even if serialized version already exists')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Instantiate and run
    HurricaneModeler(args)
