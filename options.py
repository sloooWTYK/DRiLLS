import argparse
import torch
import torch.nn as nn

class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no_cuda',
                            action='store_true',
                            default=False,
                            help='Disables CUDA training.')
        parser.add_argument('--cuda_index',
                            type=int,
                            default=0,
                            help='Cuda index you want to choose.')
        parser.add_argument('--case',
                            type=int,
                            default=3,
                            help='Case of the function')
        parser.add_argument('--dim',
                            type=int,
                            default=20,
                            help='dimension of input tensor')
        parser.add_argument('--hidden_layers',
                            type=int,
                            default=4,
                            help='number of hidden layers in Pseudo-Reversible Neural Network')
        parser.add_argument('--hidden_neurons',
                            type=int,
                            default=200,
                            help='number of neurons per hidden layer in Pseudo-Reversible Neural Network')
        parser.add_argument('--lam_adf',
                            type=float,
                            default=1.0,
                            help='weight in loss function for Active Direction Fitting loss')
        parser.add_argument('--lam_bd',
                            type=float,
                            default=1.0,
                            help='weight in loss function for Bounded Derivatives loss')
        parser.add_argument('--lr',
                            type=float,
                            default=0.001,
                            help='Initial learning rate')
        parser.add_argument('--optimizer',
                            type=str,
                            default='Adam',
                            help='Adam or LBFGS optimizer')
        parser.add_argument('--epochs_Adam',
                            type=int,
                            default=60000,
                            help='Number of epochs for Adam optimizer to train')
        parser.add_argument('--epochs_LBFGS',
                            type=int,
                            default=200,
                            help='Number of epochs for LBFGS optimizer to train')
        parser.add_argument('--TrainNum',
                            type=int,
                            default=10000,
                            help='Number of Training data')
        parser.add_argument('--ValidNum',
                            type=int,
                            default=500,
                            help='Number of Validating data')
        parser.add_argument('--TestNum',
                            type=int,
                            default=10000,
                            help='Number of Testing data')
        parser.add_argument('--coeff_para',
                            type=int,
                            default=50,
                            help='parameter for weighted NLL loss')
        parser.add_argument('--sigma',
                            type=float,
                            default=0.01,
                            help='parameter for BD loss')
        parser.add_argument('--domain',
                            nargs='+',
                            type=float,
                            default=[0.0,1.0],
                            help='Problem domain')
        parser.add_argument('--w',
                            type=list,
                            default=[0],
                            help='choice of active z in ADF loss')
        parser.add_argument('--Test_Mode',
                            type=str,
                            default='LocalFitting',
                            help='How to generate the predicted function value')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            args.device = 'cuda'
        else:
            args.device = 'cpu'
        return args
if __name__ == '__main__':
    args = Options().parse()
    print(args)