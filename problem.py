import numpy as np
import numpy.matlib as npm


def problem(x,case):
    size = x.shape
    Ns = size[0]
    dim = size[1]

    if case==1:
        f = x[:,0]**2 + x[:,1]**2
        # f1 = np.sum(np.power(x,2), axis=1)
        df = npm.repmat(np.expand_dims(f,axis=1),1,dim)
        df[:,0] = 2*x[:,0]
        df[:,1] = 2*x[:,1]

    elif case==2:
        f = 5/8*x[:,0]**2 + 5/8*x[:,1]**2 -3/4*x[:,0]*x[:,1]
        df = npm.repmat(np.expand_dims(f,axis=1),1,dim)
        df[:,0] = 5/4*x[:,0]-3/4*x[:,1]
        df[:,1] = 5/4*x[:,1]-3/4*x[:,0]

    elif case==3:
        f = x[:,0]**2 - x[:,1]**2
        df = npm.repmat(np.expand_dims(f,axis=1),1,dim)
        df[:,0] = 2*x[:,0]
        df[:,1] = -2*x[:,1]

    elif case==4:
        f = np.sum(np.power(x,2), axis=1)
        df = npm.repmat(np.expand_dims(f,axis=1),1,dim)
        for i in range(dim):
            df[:,i]=2 * x[:,i]

    elif case == 5:
        center = 0.0
        f = np.sin(np.sum((x-center) * (x-center), axis=1))
        ff  = np.cos(np.sum((x-center) * (x-center), axis=1))
        df = 2.0 * (x-center) * npm.repmat(np.expand_dims(ff, axis=1),1,dim)

    elif case == 6:
        cc = 1.
        ww = 0.0
        f = np.prod((cc**(-2.0) + (x-ww)**2.0)**(-1.0), axis=1)
        df = npm.repmat(np.expand_dims(f,axis=1),1,dim) * -1.0 * (cc**-2.0 + (x-ww)**2.0)**-1.0 * 2.0 * (x-ww)

    elif case==7:
        f = np.sum(np.power(x[:,:-1],2), axis=1) - x[:,-1]**2
        df = npm.repmat(np.expand_dims(f,axis=1),1,dim)
        df[:,:-1] = 2*x[:,:-1]
        df[:,-1] = -2*x[:,-1]

    elif case == 11:
        #thermal block case
        pass




    return f, df

def plot_z(x, y ,case):

    if case==1:
        f = x**2 + y**2
        level=[0.25,0.5,0.75,1,1.25]

    elif case==2:
        f = 5/8*x**2 + 5/8*y**2 -3/4*x*y
        level=[0.02,0.15,0.4,0.7, 1.2]

    elif case==3:
        f = x**2 - y**2
        level=[-1.25,-1,-0.75,-0.5,-0.25,0.25,0.5,0.75,1,1.25]

    return f, level