from model import EncDoc,RegNet
from dataset import Trainset, Validset, RealValidset
import numpy as np
import time
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils import save_checkpoints,setup_seed,fitFunc,fitFunc_typical, save_loss
from loss import loss_adf, loss_data, loss_bd
import scipy.optimize as syopt
import matplotlib.pyplot as plt
from problem import problem, plot_z
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import NearestNeighbors


####################Training code for NN based regression#######################
def train_reg(train_x, train_f, valid_x, valid_f, test_x, test_x_fit, epochs):
    model = RegNet(1).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = StepLR(optimizer, step_size=5000, gamma=0.7)
    train_x = train_x.detach()
    train_x = train_x.to('cuda')
    train_f = torch.unsqueeze(train_f,1)
    valid_f = torch.unsqueeze(valid_f,1)
    tt=time.time()
    for k in range(epochs):
        epoch=k
        optimizer.zero_grad()
        f=model(train_x)
        loss = torch.nn.MSELoss()(f,train_f)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (epoch+1)%100 == 0:
            model.eval()
            f_valid = model(valid_x)
            loss_valid = torch.nn.MSELoss()(f_valid,valid_f)
            print(f'#{epoch+1:5d}: valid_loss={loss_valid.item():.3e},train_loss={loss.item():.3e}, time={time.time()-tt:.2f}s')
            tt = time.time()
    print('Regression Training Finished!')
    test_f = model(test_x)
    test_f_fit = model(test_x_fit)
    return test_f, test_f_fit


class Trainer():
    def __init__(self,args):
        self.args = args
        self.device = args.device
        self.case = args.case
        self.dim = args.dim
        self.layers = args.hidden_layers
        self.neurons = args.hidden_neurons
        self.lam_adf= args.lam_adf
        self.lam_bd = args.lam_bd
        self.papercase = args.case
        self.optimizer_name = args.optimizer





        self.model_name =f'DRiLLS_ActiveDim{len(self.args.w)}_Domain_{self.args.domain}_Example{self.case}_dim{self.dim}_N{self.args.TrainNum}_' \
                         f'{self.layers}layers_{self.neurons}Neurons_' \
                         f'{self.lam_adf}adf_{self.lam_bd}bd_' \
                         f'{self.args.coeff_para}Coeff_{self.args.sigma}Sigma'

        self.EncDoc = EncDoc(self.dim, self.layers, self.neurons)
        print(self.EncDoc)

        #Generate the training data and the validation set
        self.train_set = Trainset(self.args.TrainNum, self.dim, self.args.domain, self.case)
        self.xTrain, self.fTrain, self.dfTrain = self.train_set()
        self.valid_set = Validset(self.args.ValidNum, self.dim, self.args.domain, self.case)
        self.xValid, self.fValid, self.dfValid = self.valid_set()

        ################For drawing the regression graphs##############
        self.test_set = Validset(400, self.dim, self.args.domain, self.case)
        self.xTest, self.fTest, self.dfTest   = self.test_set()

        self.xDraw, self.fDraw, _   = self.test_set()


        if self.device == 'cuda':
            self.xTrain, self.xValid, self.xTest= self.xTrain.to(self.device), self.xValid.to(self.device), self.xTest.to(self.device)
            self.fTrain, self.fValid, self.fTest= self.fTrain.to(self.device), self.fValid.to(self.device), self.fTest.to(self.device)
            self.dfTrain, self.dfValid, self.dfTest= self.dfTrain.to(self.device), self.dfValid.to(self.device), self.dfTest.to(self.device)
            self.xDraw, self.fDraw = self.xDraw.to(self.device), self.fDraw.to(self.device)

            self.EncDoc.to(self.device)

    def train(self):
        self.epochs_Adam = self.args.epochs_Adam
        self.epochs_LBFGS = self.args.epochs_LBFGS

        self.lr = self.args.lr
        self.optimizer_Adam = optim.Adam(self.EncDoc.parameters(), lr=self.lr)
        self.optimizer_LBFGS = optim.LBFGS(self.EncDoc.parameters(),max_iter=20,tolerance_grad=1e-8,tolerance_change=1e-12)
        self.lr_scheduler = StepLR(self.optimizer_Adam, step_size=5000, gamma=0.7)



        best_loss=1e10
        tt=time.time()
        self.EncDoc.train()
        step = 0
        print('Training Start...')
        if self.optimizer_name=='Adam':
            for k in range(self.epochs_Adam):
                epoch=k
                self.optimizer_Adam.zero_grad()
                self.xTrain.requires_grad_(True)

                z = self.EncDoc.encoder(self.xTrain)
                xx, jac = self.EncDoc.decoder(z)

                bd = loss_bd(jac, self.dfTrain, self.args.w, self.args.sigma)


                Loss_data = torch.nn.MSELoss()(xx, self.xTrain)
                res_adf = loss_adf(jac, self.dfTrain, self.args.w, self.args.coeff_para)
                Loss_adf = torch.nn.L1Loss()(res_adf, torch.zeros_like(res_adf))
                Loss_bd = torch.nn.L1Loss()(bd, torch.zeros_like(bd))



                Train_Loss = Loss_data + self.lam_adf * Loss_adf + self.lam_bd * Loss_bd
                Train_Loss.backward()
                self.optimizer_Adam.step()
                self.lr_scheduler.step()

                if (epoch + 1) % 100 == 0:
                    self.EncDoc.eval()
                    z = self.EncDoc.encoder(self.xValid)
                    xx, jac = self.EncDoc.decoder(z)

                    bd_valid = loss_bd(jac, self.dfValid, self.args.w, self.args.sigma)



                    Loss_data_valid = torch.nn.MSELoss()(xx, self.xValid)
                    res_adf_valid = loss_adf(jac, self.dfValid, self.args.w, self.args.coeff_para)
                    Loss_adf_valid = torch.nn.L1Loss()(res_adf_valid, torch.zeros_like(res_adf_valid))
                    Loss_bd_valid = torch.nn.L1Loss()(bd_valid, torch.zeros_like(bd_valid))


                    Valid_Loss = Loss_data_valid + self.lam_adf * Loss_adf_valid + self.lam_bd * Loss_bd_valid

                    print(
                        f'#{epoch + 1:5d}: valid_loss_data={Loss_data_valid.item():.2e},valid_loss_adf={Loss_adf_valid.item():.2e},valid_loss_bd={Loss_bd_valid.item():.2e}, loss_data={Loss_data.item():.2e}, loss_adf={Loss_adf.item():.2e},loss_bd={Loss_bd.item():.2e}, lr={self.lr_scheduler.get_last_lr()[0]:.2e}, time={time.time() - tt:.2f}s')
                    is_best = Valid_Loss < best_loss
                    state = {
                        'epoch': epoch,
                        'state_dict': self.EncDoc.state_dict(),
                        'best_loss': best_loss
                    }
                    save_checkpoints(state, is_best, save_dir=f'{self.model_name}_Adam')
                    if is_best:
                        best_loss = Valid_Loss
                    Y=np.array(f'{epoch+1:5d}, {Train_Loss:.4e}, {Loss_data:.4e}, {Loss_adf:.4e}')
                    save_loss(Y,save_dir=f'{self.model_name}_{self.optimizer_name}')
                    tt = time.time()
                if Train_Loss < 5e-5:
                    print(f'train_loss after Adam is {Train_Loss:.4e}')
                    break
            print('Training Finished!')

        if self.optimizer_name=='LBFGS':
            for k in range(self.epochs_Adam):
                epoch = k
                self.optimizer_Adam.zero_grad()
                self.xTrain.requires_grad_(True)

                z = self.EncDoc.encoder(self.xTrain)
                xx, jac = self.EncDoc.decoder(z)

                bd = loss_bd(jac, self.dfTrain, self.args.w, self.args.sigma)


                Loss_data = torch.nn.MSELoss()(xx, self.xTrain)
                res_adf = loss_adf(jac, self.dfTrain, self.args.w, self.args.coeff_para)
                Loss_adf = torch.nn.L1Loss()(res_adf, torch.zeros_like(res_adf))
                Loss_bd = torch.nn.L1Loss()(bd, torch.zeros_like(bd))


                Train_Loss = Loss_data + self.lam_adf * Loss_adf + self.lam_bd * Loss_bd
                Train_Loss.backward()
                self.optimizer_Adam.step()
                self.lr_scheduler.step()

                if (epoch + 1) % 100 == 0:
                    self.EncDoc.eval()
                    z = self.EncDoc.encoder(self.xValid)
                    xx, jac = self.EncDoc.decoder(z)

                    bd_valid = loss_bd(jac, self.dfValid, self.args.w, self.args.sigma)


                    Loss_data_valid = torch.nn.MSELoss()(xx, self.xValid)
                    res_adf_valid = loss_adf(jac, self.dfValid, self.args.w, self.args.coeff_para)
                    Loss_adf_valid = torch.nn.L1Loss()(res_adf_valid, torch.zeros_like(res_adf_valid))
                    Loss_bd_valid = torch.nn.L1Loss()(bd_valid, torch.zeros_like(bd_valid))


                    Valid_Loss = Loss_data_valid + self.lam_adf * Loss_adf_valid + self.lam_bd * Loss_bd_valid

                    print(
                        f'#{epoch + 1:5d}: valid_loss_data={Loss_data_valid.item():.2e},valid_loss_adf={Loss_adf_valid.item():.2e},valid_loss_bd={Loss_bd_valid.item():.2e}, loss_data={Loss_data.item():.2e}, loss_adf={Loss_adf.item():.2e},loss_bd={Loss_bd.item():.2e}, lr={self.lr_scheduler.get_last_lr()[0]:.2e}, time={time.time() - tt:.2f}s')
                    is_best = Valid_Loss < best_loss
                    state = {
                        'epoch': epoch,
                        'state_dict': self.EncDoc.state_dict(),
                        'best_loss': best_loss
                    }
                    save_checkpoints(state, is_best, save_dir=f'{self.model_name}_Adam')
                    if is_best:
                        best_loss = Valid_Loss
                    Y=np.array(f'{epoch+1:5d}, {Train_Loss:.4e}, {Loss_data:.4e}, {Loss_adf:.4e}')
                    save_loss(Y,save_dir=f'{self.model_name}_{self.optimizer_name}')
                    tt = time.time()
                if Train_Loss < 5e-5:
                    print(f'train_loss after Adam is {Train_Loss:.4e}')
                    break
            print('Adam Training Finished!')

            for k in range(self.epochs_LBFGS):
                epoch=k
                #########Save and Load or Not?
                #######Training the model until data loss to 1e-2 and adf loss to 5e-3 maybe?
                self.xTrain.requires_grad_(True)

                def closure():
                    self.optimizer_LBFGS.zero_grad()

                    z = self.EncDoc.encoder(self.xTrain)
                    xx, jac = self.EncDoc.decoder(z)

                    bd = loss_bd(jac, self.dfTrain, self.args.w, self.args.sigma)


                    Loss_data = torch.nn.MSELoss()(xx, self.xTrain)
                    res_adf = loss_adf(jac, self.dfTrain, self.args.w, self.args.coeff_para)
                    Loss_adf = torch.nn.L1Loss()(res_adf, torch.zeros_like(res_adf))
                    Loss_bd = torch.nn.L1Loss()(bd, torch.zeros_like(bd))


                    Train_Loss = Loss_data + self.lam_adf * Loss_adf + self.lam_bd * Loss_bd
                    Train_Loss.backward()
                    return Train_Loss

                z = self.EncDoc.encoder(self.xTrain)
                xx, jac = self.EncDoc.decoder(z)

                bd = loss_bd(jac, self.dfTrain, self.args.w, self.args.sigma)


                Loss_data = torch.nn.MSELoss()(xx, self.xTrain)
                res_adf = loss_adf(jac, self.dfTrain, self.args.w, self.args.coeff_para)
                Loss_adf = torch.nn.L1Loss()(res_adf, torch.zeros_like(res_adf))
                Loss_bd = torch.nn.L1Loss()(bd, torch.zeros_like(bd))



                self.optimizer_LBFGS.step(closure)
                Train_Loss = closure().item()


                if (epoch+1)%10 == 0:
                    self.EncDoc.eval()
                    z = self.EncDoc.encoder(self.xValid)
                    xx, jac = self.EncDoc.decoder(z)

                    bd_valid = loss_bd(jac, self.dfValid, self.args.w, self.args.sigma)


                    Loss_data_valid = torch.nn.MSELoss()(xx, self.xValid)
                    res_adf_valid = loss_adf(jac, self.dfValid, self.args.w, self.args.coeff_para)
                    Loss_adf_valid = torch.nn.L1Loss()(res_adf_valid, torch.zeros_like(res_adf_valid))
                    Loss_bd_valid = torch.nn.L1Loss()(bd_valid, torch.zeros_like(bd_valid))


                    Valid_Loss = Loss_data_valid + self.lam_adf * Loss_adf_valid + self.lam_bd * Loss_bd_valid

                    print(
                        f'#{epoch + 1:5d}: valid_loss_data={Loss_data_valid.item():.2e},valid_loss_adf={Loss_adf_valid.item():.2e},valid_loss_bd={Loss_bd_valid.item():.2e}, loss_data={Loss_data.item():.2e}, loss_adf={Loss_adf.item():.2e},loss_bd={Loss_bd.item():.2e}, lr={self.lr_scheduler.get_last_lr()[0]:.2e}, time={time.time() - tt:.2f}s')
                    is_best = Valid_Loss < best_loss
                    state = {
                        'epoch': epoch,
                        'state_dict': self.EncDoc.state_dict(),
                        'best_loss': best_loss
                    }
                    save_checkpoints(state, is_best, save_dir=f'{self.model_name}_LBFGS')
                    if is_best:
                        best_loss=Valid_Loss
                    Y=np.array(f'{epoch+1:5d}, {Train_Loss:.4e}, {Loss_data:.4e}, {Loss_adf:.4e}')
                    save_loss(Y,save_dir=f'{self.model_name}_{self.optimizer_name}')
                    tt = time.time()
                if Train_Loss < 5e-5:
                    print(f'train_loss after LBFGS is {Train_Loss:.4e}')
                    break
            print('LBFGS Training Finished!')


        print('Training Finished!')


    def test(self,degree,nestsample):

        self.EncDoc.eval()
        if self.optimizer_name=='LBFGS':
            best_model_adf = torch.load(f'checkpoints/{self.model_name}_LBFGS/checkpoint.pth.tar')
            path = f'Results/{self.papercase}/{self.model_name}_LBFGS'
        else:
            best_model_adf = torch.load(f'checkpoints/{self.model_name}_Adam/checkpoint.pth.tar')
            path = f'Results/{self.papercase}/{self.model_name}_Adam'
        self.EncDoc.load_state_dict(best_model_adf['state_dict'])

        ztrain = self.EncDoc.encoder(self.xTrain)
        zvalid = self.EncDoc.encoder(self.xValid)
        ztest = self.EncDoc.encoder(self.xTest)

        zdraw = self.EncDoc.encoder(self.xDraw)


        z_test1 = ztest.cpu().detach().numpy()[:,self.args.w].flatten()
        z_test_fit = np.linspace(np.min(z_test1),np.max(z_test1),1000)
        z_draw = zdraw.cpu().detach().numpy()[:,self.args.w].flatten()
        # z_draw_fit = np.linspace(np.min(z_draw), np.max(z_draw), np.size(z_draw))
        
        if self.args.Test_Mode == 'GlobalFitting':
            f_train=self.fTrain.cpu().detach().numpy().flatten()
            z_train = ztrain.cpu().detach().numpy()[:,self.args.w].flatten()

            params = syopt.curve_fit(fitFunc,z_train, f_train)[0]
            regParams_old = params
            f_test_pred = fitFunc(z_test_fit, *regParams_old)
            f_test_cal = fitFunc(z_test1, *regParams_old)
            f_pred = torch.from_numpy(f_test_cal).float()
            f_pred = torch.unsqueeze(f_pred,1)
            
        elif self.args.Test_Mode == 'NN':
            z_test_fit = torch.from_numpy(z_test_fit).float().to(self.device)
            z_test_fit = torch.unsqueeze(z_test_fit,1)
            f_test_cal, f_test_pred=train_reg(ztrain[:,self.args.w], self.fTrain, zvalid[:,self.args.w], self.fValid, ztest[:,self.args.w], z_test_fit, 20000)
            # f_test_cal, f_test_pred=train_reg(ztest[:,self.args.w], self.fTest, zvalid[:,self.args.w], self.fValid, ztest[:,self.args.w], z_test_fit, 20000)
            f_pred = f_test_cal.cpu()
            f_test_pred = f_test_pred.cpu().detach().numpy()
            z_test_fit = z_test_fit.cpu().detach().numpy()
            
        elif self.args.Test_Mode == 'LocalFitting':
            x_inver, _ = self.EncDoc.decoder(ztest)
            f_train = self.fTrain.cpu().detach().numpy().flatten()
            z_train = ztrain.cpu().detach().numpy()[:, self.args.w].flatten()

            nbrs = NearestNeighbors(n_neighbors=nestsample)
            nbrs.fit(self.xTrain.cpu().detach().numpy())
            idx = nbrs.kneighbors(self.xTest.cpu().detach().numpy(), return_distance=False)

            nbrs_typical = NearestNeighbors(n_neighbors=nestsample)
            nbrs_typical.fit(ztrain.cpu().detach().numpy()[:, self.args.w])
            idx_typical = nbrs_typical.kneighbors(ztest.cpu().detach().numpy()[:, self.args.w], return_distance=False)

            z_train = ztrain.cpu().detach().numpy()[:, self.args.w]
            z_test1 = ztest.cpu().detach().numpy()[:, self.args.w]
            f_test_cal = np.zeros_like(z_test1[:, 0])
            f_test_cal_typical = np.zeros_like(z_test1[:, 0])
            num = z_test1.shape[0]
            for i in range(num):
                poly = PolynomialFeatures(degree=degree)
                xps = poly.fit_transform(z_train[idx[i]])
                ls = LinearRegression()
                y = f_train[idx[i]]
                ls.fit(xps, y.reshape(-1, 1))
                xps_cal = poly.fit_transform(z_test1[[i]])
                f_test_cal[i] = ls.predict(xps_cal)

                poly_typical = PolynomialFeatures(degree=degree)
                xps_typical = poly_typical.fit_transform(z_train[idx_typical[i]])
                ls_typical = LinearRegression()
                y_typical = f_train[idx_typical[i]]
                ls_typical.fit(xps_typical, y_typical.reshape(-1, 1))
                xps_typical_cal = poly_typical.fit_transform(z_test1[[i]])
                f_test_cal_typical[i] = ls_typical.predict(xps_typical_cal)

            f_pred = torch.from_numpy(f_test_cal).float()
            f_pred = torch.unsqueeze(f_pred, 1)

            f_pred_typical = torch.from_numpy(f_test_cal_typical).float()
            f_pred_typical = torch.unsqueeze(f_pred_typical, 1)

            # #####Plotting 3D
            # if self.args.dim==2:
            #
            #     xxtest = self.xTest.cpu().detach().numpy()
            #     x_inver = x_inver.cpu().detach().numpy()
            #
            #     # syntax for 3-D plotting
            #     ax = plt.axes(projection ='3d')
            #
            #     # syntax for plotting
            #     ax.scatter(xxtest[:,0], xxtest[:,1], self.fTest.cpu().detach().numpy().flatten(),c='b')
            #     ax.scatter(x_inver[:,0], x_inver[:,1], f_test_cal,c='r')
            #     ax.set_title(r'$x^2-y^2$')
            #     plt.show()


        f = self.fTest.cpu()
        f = torch.unsqueeze(f,1)
        MSE=torch.nn.MSELoss()(f,f_pred)
        NRMSE = torch.sqrt(MSE)/(torch.max(f)-torch.min(f))
        print('DRiLLS NRMSE: ', '%.7f' %NRMSE)
        L1 = torch.nn.L1Loss()(f,f_pred) / torch.nn.L1Loss()(f,torch.zeros_like(f))
        L2 = torch.nn.MSELoss()(f,f_pred) / torch.nn.MSELoss()(f,torch.zeros_like(f))
        L2 = torch.sqrt(L2)
        print('DRiLLS L1: ', '%.7f' % L1)
        print('DRiLLS L2: ', '%.7f' % L2)

        #Typical Way of Approximation
        MSE_typical=torch.nn.MSELoss()(f,f_pred_typical)
        NRMSE_typical = torch.sqrt(MSE_typical)/(torch.max(f)-torch.min(f))
        print('Typical NRMSE: ', '%.7f' %NRMSE_typical)
        L1_typical = torch.nn.L1Loss()(f,f_pred_typical) / torch.nn.L1Loss()(f,torch.zeros_like(f))
        L2_typical = torch.nn.MSELoss()(f,f_pred_typical) / torch.nn.MSELoss()(f,torch.zeros_like(f))
        L2_typical = torch.sqrt(L2_typical)
        print('Typical L1: ', '%.7f' % L1_typical)
        print('Typical L2: ', '%.7f' % L2_typical)

        figsize = 10, 10
        figure, ax = plt.subplots(figsize=figsize)
        plt.tick_params(labelsize=15)
        font2 = {'weight': 'normal',
                 'size': 25,
                 }
        plt.xlabel('z1', font2)
        plt.plot(z_draw, self.fDraw.cpu().detach().numpy().flatten(), 'bo', label='Exact Value')
        plt.plot(z_draw, f_pred, 'ro', label='Predicted Value')
        plt.legend(fontsize=30)



        path = f'{path}/{self.args.Test_Mode}'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}/DRiLLS_Regression.png',bbox_inches='tight',pad_inches=0.2)
        ErrorFile = f"{path}/DRiLLS_error.txt"
        with open(ErrorFile, 'w') as ff:
            ff.write(str('Local Fitting:\n'))
            ff.write(str(f'NRMSE:{NRMSE:.4f},RL1:{L1:.4f},RL2:{L2:.4f}\n'))
            ff.write(str('Typical Fitting:\n'))
            ff.write(str(f'NRMSE:{NRMSE_typical:.4f},RL1:{L1_typical:.4f},RL2:{L2_typical:.4f}'))
            ff.close()
        plt.draw()
        plt.clf()



#################2d Quiver and Level Sets Graphs####################
        if self.dim==2:
            x = np.linspace(-1, 1, 15)
            y = np.linspace(-1, 1, 15)
            x_, y_ = np.meshgrid(x, y)
            xxxx = np.hstack((x_.reshape(-1, 1), y_.reshape(-1, 1)))
            xxxx = torch.from_numpy(xxxx).float().to(self.device)
            zzzz = self.EncDoc.encoder(xxxx)
            _,jac = self.EncDoc.decoder(zzzz)

            fTest,dfTest = problem(xxxx.cpu().detach().numpy(),self.case)
            fTest = torch.from_numpy(fTest).float()
            dfTest = torch.from_numpy(dfTest).float()

            fTest, dfTest = fTest.to(self.device), dfTest.to(self.device)


            plt.quiver(xxxx.cpu().detach().numpy()[:, 0], xxxx.cpu().detach().numpy()[:, 1],
                       dfTest.cpu().detach().numpy()[:, 0], dfTest.cpu().detach().numpy()[:, 1], color='blue', width=0.0025)
            plt.quiver(xxxx.cpu().detach().numpy()[:, 0], xxxx.cpu().detach().numpy()[:, 1],
                       jac[:,:,1].cpu().detach().numpy()[:, 0], jac[:,:,1].cpu().detach().numpy()[:, 1], color='red', width=0.0025)

            x = y = np.linspace(-1, 1, 100)

            x, y = np.meshgrid(x,y)
            z,level=plot_z(x,y,self.case)
            plt.contour(x, y, z, level,colors='#D0D3D4')
            plt.xlabel('x1',font2)
            plt.ylabel('x2',font2)
            plt.axis('square')
            plt.axis([-1.2, 1.2, -1.2, 1.2])

            plt.savefig(f'{path}/DRiLLS_Quiver.png',bbox_inches='tight',pad_inches=0.2)


class Tester():
    def __init__(self,args):
        self.args = args
        self.device = args.device
        self.case = args.case
        self.dim = args.dim
        self.layers = args.hidden_layers
        self.neurons = args.hidden_neurons
        self.lam_adf= args.lam_adf
        self.lam_bd = args.lam_bd
        self.papercase = args.case
        self.optimizer_name = args.optimizer




        self.model_name =f'DRiLLS_ActiveDim{len(self.args.w)}_Domain_{self.args.domain}_Example{self.case}_dim{self.dim}_N{self.args.TrainNum}_' \
                         f'{self.layers}layers_{self.neurons}Neurons_' \
                         f'{self.lam_adf}adf_{self.lam_bd}bd_' \
                         f'{self.args.coeff_para}Coeff_{self.args.sigma}Sigma'

        self.EncDoc = EncDoc(self.dim, self.layers, self.neurons)
        print(self.EncDoc)

        #Generate the testing data
        self.train_set = Trainset(self.args.TrainNum, self.dim, self.args.domain, self.case)
        self.xTrain, self.fTrain, self.dfTrain = self.train_set()



        if self.device == 'cuda':
            self.xTrain, self.fTrain, self.dfTrain = self.xTrain.to(self.device), self.fTrain.to(self.device), self.dfTrain.to(self.device)
            self.EncDoc.to(self.device)


#################Numerical Error Testing##############
    def TrueTest(self,degree,NumofIteration):

        self.EncDoc.eval()
        if self.optimizer_name=='LBFGS':
            best_model_adf = torch.load(f'checkpoints/{self.model_name}_LBFGS/checkpoint.pth.tar')
            path = f'Results/{self.papercase}/{self.model_name}_LBFGS'
        else:
            best_model_adf = torch.load(f'checkpoints/{self.model_name}_Adam/checkpoint.pth.tar')
            path = f'Results/{self.papercase}/{self.model_name}_Adam'
        self.EncDoc.load_state_dict(best_model_adf['state_dict'])

        L1_all=0
        L2_all=0
        NRMSE_all=0
        L1_typical_all=0
        L2_typical_all=0
        NRMSE_typical_all=0
        ztrain = self.EncDoc.encoder(self.xTrain)

        path = f'{path}/{self.args.Test_Mode}'
        if not os.path.exists(path):
            os.makedirs(path)
        ErrorFile = f"{path}/DRiLLS_error_degree{degree}.txt"

        for numi in range(NumofIteration):
            self.temp_testset = RealValidset(self.args.TestNum, self.dim, self.args.domain, self.case)
            self.xTest, self.fTest, self.dfTest = self.temp_testset(numi)
            self.xTest, self.fTest, self.dfTest = self.xTest.to(self.device), self.fTest.to(self.device), self.dfTest.to(self.device)
            self.xValid, self.fValid = torch.clone(self.xTest), torch.clone(self.fTest)

            print(f'Iteration_{numi}:')

            zvalid = self.EncDoc.encoder(self.xTest)
            ztest = self.EncDoc.encoder(self.xTest)


            z_test1 = ztest.cpu().detach().numpy()[:,self.args.w].flatten()
            z_test_fit = np.linspace(np.min(z_test1),np.max(z_test1),1000)

            if self.args.Test_Mode == 'GlobalFitting':
                f_train=self.fTrain.cpu().detach().numpy().flatten()
                z_train = ztrain.cpu().detach().numpy()[:,self.args.w].flatten()

                params = syopt.curve_fit(fitFunc,z_train, f_train)[0]
                regParams_old = params
                f_test_pred = fitFunc(z_test_fit, *regParams_old)
                f_test_cal = fitFunc(z_test1, *regParams_old)
                f_pred = torch.from_numpy(f_test_cal).float()
                f_pred = torch.unsqueeze(f_pred,1)
                f_pred_typical = torch.clone(f_pred)
                
            elif self.args.Test_Mode == 'NN':
                z_test_fit = torch.from_numpy(z_test_fit).float().to(self.device)
                z_test_fit = torch.unsqueeze(z_test_fit,1)
                f_test_cal, f_test_pred=train_reg(ztrain[:,self.args.w], self.fTrain, zvalid[:,self.args.w], self.fValid, ztest[:,self.args.w], z_test_fit, 20000)
                # f_test_cal, f_test_pred=train_reg(ztest[:,self.args.w], self.fTest, zvalid[:,self.args.w], self.fValid, ztest[:,self.args.w], z_test_fit, 20000)
                f_pred = f_test_cal.cpu()
                f_pred_typical = torch.clone(f_pred)
                f_test_pred = f_test_pred.cpu().detach().numpy()
                z_test_fit = z_test_fit.cpu().detach().numpy()

            elif self.args.Test_Mode == 'LocalFitting':
                x_inver,_ = self.EncDoc.decoder(ztest)
                f_train=self.fTrain.cpu().detach().numpy().flatten()
                z_train = ztrain.cpu().detach().numpy()[:,self.args.w].flatten()

                nbrs = NearestNeighbors(n_neighbors=30)
                nbrs.fit(self.xTrain.cpu().detach().numpy())
                idx = nbrs.kneighbors(self.xTest.cpu().detach().numpy(), return_distance=False)

                nbrs_typical = NearestNeighbors(n_neighbors=30)
                nbrs_typical.fit(ztrain.cpu().detach().numpy()[:,self.args.w])
                idx_typical = nbrs_typical.kneighbors(ztest.cpu().detach().numpy()[:,self.args.w], return_distance=False)

                z_train = ztrain.cpu().detach().numpy()[:,self.args.w]
                z_test1 = ztest.cpu().detach().numpy()[:,self.args.w]
                f_test_cal = np.zeros_like(z_test1[:, 0])
                f_test_cal_typical = np.zeros_like(z_test1[:, 0])
                num = z_test1.shape[0]
                for i in range(num):
                    poly = PolynomialFeatures(degree=degree)
                    xps = poly.fit_transform(z_train[idx[i]])
                    ls = LinearRegression()
                    y = f_train[idx[i]]
                    ls.fit(xps, y.reshape(-1, 1))
                    xps_cal = poly.fit_transform(z_test1[[i]])
                    f_test_cal[i] = ls.predict(xps_cal)

                    poly_typical = PolynomialFeatures(degree=degree)
                    xps_typical = poly_typical.fit_transform(z_train[idx_typical[i]])
                    ls_typical = LinearRegression()
                    y_typical = f_train[idx_typical[i]]
                    ls_typical.fit(xps_typical, y_typical.reshape(-1, 1))
                    xps_typical_cal = poly_typical.fit_transform(z_test1[[i]])
                    f_test_cal_typical[i] = ls_typical.predict(xps_typical_cal)


                f_pred = torch.from_numpy(f_test_cal).float()
                f_pred = torch.unsqueeze(f_pred,1)

                f_pred_typical = torch.from_numpy(f_test_cal_typical).float()
                f_pred_typical = torch.unsqueeze(f_pred_typical,1)


            f = self.fTest.cpu()
            f = torch.unsqueeze(f,1)
            MSE=torch.nn.MSELoss()(f,f_pred)
            NRMSE = torch.sqrt(MSE)/(torch.max(f)-torch.min(f))
            print('DRiLLS NRMSE: ', '%.7f' %NRMSE)
            L1 = torch.nn.L1Loss()(f,f_pred) / torch.nn.L1Loss()(f,torch.zeros_like(f))
            L2 = torch.nn.MSELoss()(f,f_pred) / torch.nn.MSELoss()(f,torch.zeros_like(f))
            L2 = torch.sqrt(L2)
            print('DRiLLS L2: ', '%.7f' % L2)
            print('DRiLLS L1: ', '%.7f' % L1)

            #Typical Way of Approximation
            MSE_typical=torch.nn.MSELoss()(f,f_pred_typical)
            NRMSE_typical = torch.sqrt(MSE_typical)/(torch.max(f)-torch.min(f))
            print('Typical NRMSE: ', '%.7f' %NRMSE_typical)
            L1_typical = torch.nn.L1Loss()(f,f_pred_typical) / torch.nn.L1Loss()(f,torch.zeros_like(f))
            L2_typical = torch.nn.MSELoss()(f,f_pred_typical) / torch.nn.MSELoss()(f,torch.zeros_like(f))
            L2_typical = torch.sqrt(L2_typical)
            print('Typical L2: ', '%.7f' % L2_typical)
            print('Typical L1: ', '%.7f' % L1_typical)
            print('\n')




            if numi == 0:
                with open(ErrorFile, 'w') as ff:
                    ff.write(str(f'{numi}_Local Fitting:\n'))
                    ff.write(str(f'NRMSE:{NRMSE:.4f},RL1:{L1:.4f},RL2:{L2:.4f}\n'))
                    ff.write(str(f'{numi}_Typical Fitting:\n'))
                    ff.write(str(f'NRMSE:{NRMSE_typical:.4f},RL1:{L1_typical:.4f},RL2:{L2_typical:.4f}\n'))
                    ff.write(str('\n'))
                    ff.close()
            else:

                with open(ErrorFile, 'a') as ff:
                    ff.write(str(f'{numi}_Local Fitting:\n'))
                    ff.write(str(f'NRMSE:{NRMSE:.4f},RL1:{L1:.4f},RL2:{L2:.4f}\n'))
                    ff.write(str(f'{numi}_Typical Fitting:\n'))
                    ff.write(str(f'NRMSE:{NRMSE_typical:.4f},RL1:{L1_typical:.4f},RL2:{L2_typical:.4f}\n'))
                    ff.write(str('\n'))
                    ff.close()

            L1_all = L1_all + L1
            L2_all = L2_all + L2
            NRMSE_all = NRMSE_all + NRMSE
            L1_typical_all = L1_typical_all + L1_typical
            L2_typical_all = L2_typical_all + L2_typical
            NRMSE_typical_all = NRMSE_typical_all + NRMSE_typical

        ErrorFile_final = f"{path}/DRiLLS_error_final_degree{degree}.txt"
        with open(ErrorFile_final, 'w') as ff:
            ff.write(str(f'Final_Local Fitting:\n'))
            ff.write(str(f'NRMSE:{NRMSE_all/NumofIteration:.4f},,RL1:{L1_all/NumofIteration:.4f},RL2:{L2_all/NumofIteration:.4f}\n'))
            ff.write(str(f'Final_Typical Fitting:\n'))
            ff.write(str(f'NRMSE:{NRMSE_typical_all/NumofIteration:.4f},RL1:{L1_typical_all/NumofIteration:.4f},RL2:{L2_typical_all/NumofIteration:.4f}\n'))
            ff.close()


################Sensitivity Testing##################
    def SensitivityTest(self,NumofIteration):

        self.EncDoc.eval()
        if self.optimizer_name=='LBFGS':
            best_model_adf = torch.load(f'checkpoints/{self.model_name}_LBFGS/checkpoint.pth.tar')
            path = f'Results/{self.papercase}/{self.model_name}_LBFGS'
        else:
            best_model_adf = torch.load(f'checkpoints/{self.model_name}_Adam/checkpoint.pth.tar')
            path = f'Results/{self.papercase}/{self.model_name}_Adam'
        self.EncDoc.load_state_dict(best_model_adf['state_dict'])

        Sens_x_all=torch.zeros(self.dim)
        Sens_z_all=torch.zeros(self.dim)


        path = f'{path}'
        if not os.path.exists(path):
            os.makedirs(path)
        SensFile = f"{path}/sensitivity.txt"

        for numi in range(NumofIteration):
            self.temp_testset = RealValidset(self.args.TestNum, self.dim, self.args.domain, self.case)
            self.xTest, self.fTest, self.dfTest = self.temp_testset(numi)
            self.xTest, self.fTest, self.dfTest = self.xTest.to(self.device), self.fTest.to(self.device), self.dfTest.to(self.device)


            ztest = self.EncDoc.encoder(self.xTest)

            x_inv, jac = self.EncDoc.decoder(ztest)
            self.xTest = self.xTest.cpu().detach()
            self.dfTest = self.dfTest.cpu().detach()
            jac = jac.cpu().detach()



            avg_dfdx = torch.sum(self.dfTest, 0) / len(self.dfTest)

            dfdz = torch.matmul(self.dfTest.unsqueeze(1), jac).squeeze(1)
            avg_dfdz = torch.sum(dfdz, 0) / len(self.xTest)

            sens_x = torch.abs(avg_dfdx) / torch.sum(torch.abs(avg_dfdx))
            sens_z = torch.abs(avg_dfdz) / torch.sum(torch.abs(avg_dfdz))
            print(f'Iteration_{numi}:')




            if numi == 0:
                with open(SensFile, 'w') as ff:
                    ff.write(str(f'{numi}_Sensitivity:\n'))
                    ff.write(str(f'Sens_z:{sens_z}\n'))
                    ff.write(str(f'Sens_x:{sens_x}\n'))
                    ff.write(str('\n'))
                    ff.close()
            else:

                with open(SensFile, 'a') as ff:
                    ff.write(str(f'{numi}_Sensitivity:\n'))
                    ff.write(str(f'Sens_z:{sens_z}\n'))
                    ff.write(str(f'Sens_x:{sens_x}\n'))
                    ff.write(str('\n'))
                    ff.close()

            Sens_x_all = Sens_x_all + sens_x
            Sens_z_all = Sens_z_all + sens_z

        SensFile_final = f"{path}/sensitivity_final.txt"
        with open(SensFile_final, 'w') as ff:
            ff.write(str(f'Final_Sensitivity:\n'))
            ff.write(str(f'Sens_z:{Sens_z_all/NumofIteration}\n'))
            ff.write(str(f'Sens_x:{Sens_x_all/NumofIteration}\n'))
            ff.close()

