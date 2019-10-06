import argparse
from utils import str2bool

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()

    #experiment specific parameters
    parser.add_argument('--data_volume', nargs="?", type=int, default=50000, help='Data Volume for all of the experiment')
    # parser.add_argument('--vector_type', nargs="?", type=int, default=0, help='Types of vectors to be constructed from csv data.')
    parser.add_argument('--suffix', nargs="?", type=str, default="50K",
                        help='Default suffix used to store the related files')
    # parser.add_argument('--survey_strat', nargs="?", type=str, default="both",
    #                     help='survey strategy: can be WDF DDF or both')

    args = parser.parse_args()
    return args