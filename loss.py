import torch


######active direction fitting loss#########
def loss_adf(jac, df, w, coeff_para):
    df_norm = torch.linalg.norm(df,ord=2,axis=1)
    coeff = 1+coeff_para*torch.exp(-df_norm)
    df = torch.unsqueeze(df, 2)
    jac_df = torch.sum(jac*df,1)
    jac_df[:,w] =0
    res = torch.sum(jac_df * jac_df, 1)
    res_adf = coeff*res
    return res_adf


#############data fitting loss#######
def loss_data(x,x_bar):
    return  torch.nn.MSELoss()(x,x_bar)


###########bounded derivatives loss############
def loss_bd(jac, df, w, sigma):
    df = torch.unsqueeze(df, 2)
    jac_df = torch.sum(jac*df,1)
    j_active = jac_df[:,w]
    active_norm=torch.linalg.norm(j_active, ord=2, axis=1).unsqueeze(-1)
    bd = torch.sigmoid((active_norm-1)/sigma)
    return bd
