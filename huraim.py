import getopt
import sys


class HurricaneModeler:

    def __init__(self, **kwargs):
        self.input_data = kwargs['input_data'] if 'input_data' in kwargs.keys() else 'data/hurdat2.txt'
        self.error_data = kwargs[
            'error_data'] if 'error_data' in kwargs.keys() else 'errors/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt'
        self.retrain = kwargs['retrain'] if 'retrain' in kwargs.keys() else False


def parse_args(argv):
    # Set defaults
    input_data = 'data/hurdat2.txt'
    error_data = 'errors/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt'

    # Read in command-line options
    options, _ = getopt.getopt(argv, 'i:r', ['input_data=', 'retrain'])

    for opt, arg in options:
        if opt in ('i', '--input_data='):
            input_data = arg
        elif opt in ('e', '--error_data='):
            error_data = arg
        elif opt in ('-r', '--retrain'):
            retrain = True

    return {'input_data': input_data, 'error_data': error_data, 'retrain': retrain}


if __name__ == '__main__':
    # Parse arguments
    args = parse_args(sys.argv[1:])

    # Instantiate and run
    HurricaneModeler(**args)
