import argparse
from pathlib import Path



class ArgumentLoader(object):
    def parse_args(self):
        args = self.get_parser().parse_args()
        return args


    def get_parser(self):
        parser = argparse.ArgumentParser()

        # mode options
        parser.add_argument('--quiet', '-q', action='store_true', help='Do not output log file')

        # gpu options
        # parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device id (use CPU if specify a negative value)')
        parser.add_argument('--cuda', action='store_true', help='Use CUDA')

        # training parameters
        parser.add_argument('--epoch', '-e', type=int, default=10, help='Conduct training up to i-th epoch (Default: 10)')
        parser.add_argument('--embed_dim', type=int, default=100, help='The number of embedding dimension (Default: 100)')
        parser.add_argument('--num_class', type=int, default=5, help='The number of class(es) (Default: 5)')
        parser.add_argument('--max_norm', type=float, default=5, help='Max norm of the gradients for Clips gradient norm of an iterable of parameters (Default: 5)')
        parser.add_argument('--norm_type', type=float, default=2, help='Type of the used p-norm for Clips gradient norm of an iterable of parameters (Default: 2)')

        # optimizer parameters
        parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='Initial learning rate for SGD (Default: 0.1)')
        parser.add_argument('--sgd_momentum_ratio', dest='momentum', type=float, default=0.9, help='Momentum ratio for SGD (Default:0.9)')

        # data paths and related options
        parser.add_argument('--train_data', type=Path, default=None, help='File path to training data')
        parser.add_argument('--valid_data', type=Path, default=None, help='File path to validation data')
        parser.add_argument('--test_data', type=Path, default=None, help='File path to test data')
        parser.add_argument('--vocab_indices', type=Path, default=None, help='File path to vocab indices mapping')
        parser.add_argument('--output_data', type=Path, default=None, help='File path to output file')

        return parser
