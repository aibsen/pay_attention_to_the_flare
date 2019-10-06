import argparse
from utils import str2bool

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    defaultDir = '../data'

    defaultTrainingSet = defaultDir+"/train_data_interpolated1K.h5"
    defaultValidationSet = defaultDir+"/val_data_interpolated1K.h5"
    defaultTestSet = defaultDir+"/test_data_interpolated1K.h5"

    parser = argparse.ArgumentParser()

    #experiment specific parameters
    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Batch_size for experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=1772670,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=30, help='The experiment\'s epoch budget')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=True,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--learning_rate', nargs="?", type=float, default=1e-03,
                        help='learning rate to use for Adam')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=0,
                        help='Weight decay to use for Adam')
    parser.add_argument('--train_data', nargs="?", type=str, default=defaultTrainingSet, help='.h5 with train set, X, Y and ids')
    parser.add_argument('--val_data', nargs="?", type=str, default=defaultValidationSet, help='.h5 with val set, X, Y and ids')
    parser.add_argument('--test_data', nargs="?", type=str, default=defaultTestSet, help='.h5 with test set, X, Y and ids')
    parser.add_argument('--num_output_classes', nargs="?", type=int, default=15, help='Number of posssible classes')

    args = parser.parse_args()
    return args