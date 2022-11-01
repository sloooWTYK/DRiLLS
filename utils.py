import torch
import numpy as np
import os
import shutil




def fitFunc(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4

def fitFunc_typical(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoints(state, is_best=None,
                     base_dir='checkpoints',
                     save_dir=None):
    if save_dir:
        save_dir = os.path.join(base_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    checkpoint = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth.tar')
        shutil.copyfile(checkpoint, best_model)

def save_loss(state,base_dir='Loss',save_dir=None):
    if save_dir:
        save_dir = os.path.join(base_dir, save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    loss_file = os.path.join(save_dir, 'loss.txt')
    with open(loss_file, 'a') as ff:
        ff.write(str(state))
        ff.write('\n')
        ff.close()

def jacobian(output, input):
    n, dim = input.shape
    w = torch.ones_like(input[:,[0]])

    jacob = torch.empty(dim,n,dim).to(output.device)
    for i in range(dim):
        output_i = output[:,[i]]
        jacob[i] = torch.autograd.grad(output_i, input, w, create_graph=True)[0]
    jacob=jacob.permute(1,0,2)
    return jacob


