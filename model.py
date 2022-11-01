import torch
import torch.nn as nn
from utils import jacobian

class EncDoc(nn.Module):
    def __init__(self,dim, hidden_layers, hidden_neurons):
        super().__init__()
        self.hidden_layers=hidden_layers
        self.hidden_neurons = hidden_neurons
        self.dim =dim

        enc_1 = nn.Linear(self.dim,self.hidden_neurons)
        setattr(self, f'enc{1}',enc_1)

        for i in range(1, self.hidden_layers):
            enc = nn.Linear(self.hidden_neurons, self.hidden_neurons)
            setattr(self, f'enc{i+1}', enc)

        enc_last = nn.Linear(self.hidden_neurons, self.dim)
        setattr(self, f'enc_last',enc_last)

        dec_1 = nn.Linear(self.dim,self.hidden_neurons)
        setattr(self, f'dec{1}',dec_1)

        for i in range(1, self.hidden_layers):
            dec = nn.Linear(self.hidden_neurons, self.hidden_neurons)
            setattr(self, f'dec{i+1}', dec)

        dec_last = nn.Linear(self.hidden_neurons, self.dim)
        setattr(self, f'dec_last',dec_last)



    def forward(self,x):
        xx=torch.clone(x)
        for i in range(self.hidden_layers):
            xx = torch.tanh(getattr(self, f'enc{i+1}')(xx))
        xx = getattr(self, f'enc_last')(xx)
        zzz = torch.clone(xx)
        for i in range(self.hidden_layers):
            xx = torch.tanh(getattr(self, f'dec{i+1}')(xx))
        xx = getattr(self, f'dec_last')(xx)
        return zzz,xx

    def encoder(self,x):
        xx=torch.clone(x)
        for i in range(self.hidden_layers):
            xx = torch.tanh(getattr(self, f'enc{i+1}')(xx))
        xx = getattr(self, f'enc_last')(xx)
        zzz = torch.clone(xx)
        return zzz

    def decoder(self,zzz):
        zzz.requires_grad_(True)
        xx=torch.clone(zzz)
        for i in range(self.hidden_layers):
            xx = torch.tanh(getattr(self, f'dec{i+1}')(xx))
        xx = getattr(self, f'dec_last')(xx)

        jac = jacobian(xx,zzz)
        return xx,jac

    def encoder_grad(self,x):
        x.requires_grad_(True)
        xx=torch.clone(x)
        for i in range(self.hidden_layers):
            xx = torch.tanh(getattr(self, f'enc{i+1}')(xx))
        xx = getattr(self, f'enc_last')(xx)
        jac = jacobian(xx,x)
        return jac

###############A simple 2 layer NN for regression###########
class RegNet(nn.Module):
    def __init__(self,dim):
        super().__init__()


        self.fc_1 = nn.Linear(dim,10)
        self.fc_2 = nn.Linear(10,1)

    def forward(self, z):
        xx=torch.clone(z)
        xx=torch.tanh(self.fc_1(xx))
        xx=self.fc_2(xx)
        return xx



if __name__ == '__main__':
    dim=2
    x =torch.randn(10,dim).float()
    net = EncDoc(dim, 3,10)
    for name,para in net.named_parameters():
        print(name)

    a,b=net(x)
    zz=net.encoder(x)
    xx,dx=net.decoder(zz)
