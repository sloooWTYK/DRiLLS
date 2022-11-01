import torch
import numpy as np
from pyDOE import lhs
from problem import problem
from scipy import io
from utils import setup_seed
class Trainset(object):
    def __init__(self, NumTrain, Npar, domain, case):
        self.Npar = Npar
        self.NumTrain = NumTrain
        self.lb = domain[0] * np.ones(self.Npar)
        self.ub = domain[1] * np.ones(self.Npar)
        self.case = case




    def __call__(self):
###############Here I manually fix the random seed for training samples###############
        setup_seed(666)
#############for PDE case ################
        if self.case==11:
            mat = io.loadmat(f'data/{self.Npar}_{self.NumTrain}.mat')
            self.xTrain, self.fTrain, self.dfTrain=mat['sample_set'],mat['f_data'],mat['f_deri']
            self.fTrain=np.squeeze(self.fTrain)
        else:
            self.xTrain = self.lb + (self.ub - self.lb) * lhs(self.Npar, self.NumTrain)
            self.fTrain,self.dfTrain = problem(self.xTrain,self.case)
        self.xTrain = torch.from_numpy(self.xTrain).float()
        self.fTrain = torch.from_numpy(self.fTrain).float()
        self.dfTrain = torch.from_numpy(self.dfTrain).float()

        return self.xTrain, self.fTrain, self.dfTrain


class Validset(object):
    def __init__(self, NumValid, Npar, domain, case):
        self.Npar = Npar
        self.NumValid = NumValid
        self.lb = domain[0] * np.ones(self.Npar)
        self.ub = domain[1] * np.ones(self.Npar)
        self.case = case


    def __call__(self):
        setup_seed(66)
        if self.case==11:
            mat = io.loadmat(f'data/{self.Npar}_{self.NumValid}_test.mat')
            self.xValid, self.fValid, self.dfValid=mat['sample_set'],mat['f_data'],mat['f_deri']
            self.fValid=np.squeeze(self.fValid)
        else:
            self.xValid = self.lb + (self.ub - self.lb) * lhs(self.Npar, self.NumValid)
            self.fValid,self.dfValid = problem(self.xValid,self.case)
        self.xValid = torch.from_numpy(self.xValid).float()
        self.fValid = torch.from_numpy(self.fValid).float()
        self.dfValid = torch.from_numpy(self.dfValid).float()

        return self.xValid, self.fValid, self.dfValid

class RealValidset(object):
    def __init__(self, NumValid, Npar, domain, case):
        self.Npar = Npar
        self.NumValid = NumValid
        self.lb = domain[0] * np.ones(self.Npar)
        self.ub = domain[1] * np.ones(self.Npar)
        self.case = case



    def __call__(self,index):
        if self.case==11:
            mat = io.loadmat(f'data/test/{self.Npar}_{self.NumValid}_test_{index+1}.mat')
            self.xValid, self.fValid, self.dfValid=mat['sample_set'],mat['f_data'],mat['f_deri']
            self.fValid=np.squeeze(self.fValid)
        else:
            self.xValid = self.lb + (self.ub - self.lb) * lhs(self.Npar, self.NumValid)
            self.fValid,self.dfValid = problem(self.xValid,self.case)
        self.xValid = torch.from_numpy(self.xValid).float()
        self.fValid = torch.from_numpy(self.fValid).float()
        self.dfValid = torch.from_numpy(self.dfValid).float()
        return self.xValid, self.fValid, self.dfValid


if __name__ == '__main__':
    train_set = Trainset(100, 4, (-1,1),11)
    x_train,f_train, df_train = train_set()

    valid_set = Validset(100, 2)
    x_valid = valid_set()
    print(train_set)
    print(x_train)
    print(x_valid)