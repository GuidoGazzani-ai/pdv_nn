# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:46:16 2023

@author: Guido Gazzani
"""


import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import pickle
import keras
import pybobyqa
from scipy.optimize import minimize
import time
from tqdm import tqdm


### my modules
from sampler_PDV_bnp import *
from pricing_vix_neural_bnp import *
from generate_parameters_PDV import *
from plot_save_results_calib_bnp import return_joint_results, return_spx_results
from times_to_maturity import maturities_joint_func, extract_monthly_VIX, extract_monthly_SPX 


### my flags

yfinance=False
flag_cluster=True
flag_save_plots=True
flag_load_initial_guess=True
flag_run_pricing_examples=True

flag_only_vix=False
flag_only_spx=False
flag_joint=True

variance_reduction=False

scaling_maturities=365

### my specifications
user='ag_cu-student'
name_network='model_NN_batch64-7Bayesian'

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dt = 1/252   ## 1 business day
timestep_per_day=2

flag_only_monthly=True

OPTIMIZERS=['PYBOBYQA','SLSQP','L-BFGS-B','TNC']
OPTIMIZER=OPTIMIZERS[0]

    
def restructure_date(day):
    '''Needed '''
    n=2
    list_=[day[1:][i:i+n] for i in range(0, len(day), n)]
    return list_[0]+list_[1]+'-'+list_[2]+'-'+list_[3]

#day=r'/20210427'
#day=r'/20210602'
day=r'/20210602'
#day=r'/20231025'

day_restructured=restructure_date(day)

if flag_only_monthly:
    maturities_spx=extract_monthly_SPX(day_restructured)
    maturities_vix=extract_monthly_VIX(day_restructured)[:3]    
    maturities_joint=maturities_joint_func(maturities_vix,maturities_spx)
    maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
    
else:
    ## to be adjusted
        
    maturities_vix=[r'13.0days',r'48.0days',r'76.0days',r'104.0days',r'139.0days',r'167.0days']
    maturities_spx=[r'15.0days',r'29.0days',r'43.0days',r'50.0days',r'78.0days',r'134.0days',r'169.0days',r'180.0days',r'288.0days',r'351.0days']
    maturities_joint=[r'13.0days',r'15.0days',r'29.0days',r'48.0days',r'50.0days',r'76.0days',r'78.0days',r'104.0days',r'134.0days',r'139.0days',r'169.0days',r'167.0days',r'180.0days',r'288.0days',r'351.0days']
    maturities_joint_labels=[string.split('.')[0]+' '+string[-4:] for string in maturities_joint]
    
    

if flag_cluster:### change the dirs
    dir_plots_cluster=r'/home/'+user+'/PDV/bnp/data'+day
    dir_calibration=r'/home/'+user+'/PDV/bnp/calibration'+day
    dir_calibration_monthly=r'/home/'+user+'/PDV/bnp/calibration'+day+'/only_monthly'
    os.makedirs(dir_plots_cluster,exist_ok=True)
    os.makedirs(dir_calibration_monthly,exist_ok=True)
else:
    dir_data=r'C:\Users\Guido Gazzani\Desktop\Paris\PDV\cluster_codes\bnp_data'+'\\'+day[1:]


os.chdir(dir_plots_cluster) #########################################


print('DIRECTORY OF DATA:', dir_plots_cluster)

with open('dict_data_for_calib_bnp_spx_restricted.pkl', 'rb') as f:
    loaded_dict_spx = pickle.load(f)
print('Successfully loaded dict with SPX options_data')
    
with open('dict_data_for_calib_bnp_vix_restricted.pkl', 'rb') as f:
    loaded_dict_vix = pickle.load(f)
print('Successfully loaded dict with VIX options_data')



####################### EXAMPLE PRICING SPX

lam1 = torch.tensor([55,10])
lam2 = torch.tensor([20 ,3])
betas = torch.tensor([0.04, -0.13, 0.6]) # beta_0, beta_1, beta_2 . beta_12 is in `parabolic` variable
theta1 = torch.tensor(0.2)
theta2 = torch.tensor(0.85)
parabolic = torch.tensor(0.1)  # beta_12
parabolic_offset = torch.tensor(0)  # c
flag_plot=True
S_0=torch.tensor(1.0)



if yfinance:
    import yfinance as yf
    from datetime import timedelta
    'Unfortunately the package yfinance does not run the cluster; dependency problem - One can also use Rstudio' 
    asofdate = pd.to_datetime(day_restructured)
    load_from = asofdate - timedelta(days=1000)  # Use the past 4 years to initialize
    spx = yf.Ticker("^GSPC").history(start=load_from, end=asofdate + timedelta(days=1))['Close']
    spx.index = pd.to_datetime(spx.index.date)
else:
    from datetime import timedelta        
    dir_past_returns=r'/home/ag_cu-student/initial_values/codes_for_OM_data'
    os.chdir(dir_past_returns)
    with open('dictionary_initial_values_total.pkl', 'rb') as f:
        init_dict = pickle.load(f)
    spx=init_dict[day_restructured]
    
def initialize_R(lam, dt,past_prices=None, max_delta=1000, transform=identity):
    """
    Initialize the R_j for the 4FMPDV model.
    :param lam: torch.tensor of size 2. contains \lambda_{j,1} and \lambda_{j,2}
    :param past_prices: pd.Series. Contains the past prices of the asset.
    :param max_delta: int. Number of past returns to use to compute the weighted averages.
    :param transform: should be identity for R_1 and squared for R_2.
    :return: torch.tensor of size 2. containing R_j1 and R_j2
    """
    #returns = (past_prices - past_prices.shift(1)) / past_prices.shift(1)
    returns = 1 - past_prices.shift(1) / past_prices
    returns = torch.tensor(returns.values, device=lam.device)[-max_delta:].flip(0)
    timestamps = torch.arange(returns.shape[0], device=lam.device) * dt
    timestamps.unsqueeze_(0)
    lam = lam.unsqueeze(1)
    weights = exp_kernel_GPU(timestamps, lam)
    x = transform(returns)
    return torch.sum(x * weights, dim=1)

dt=1/252
R_init1 = initialize_R(lam1, dt, past_prices=spx, transform=identity)
R_init2 = initialize_R(lam2, dt, past_prices=spx, transform=squared)


print('R_init1:',R_init1)
print('R_init2:',R_init2)

N = 10 # number of paths 
timestep_per_day = 2 # number of timestep per business day
maturity = 1 # maturity in years.
# Input vix_N not needed here as we do not rely on Nested Monte Carlo for the VIX
# Simulation. 
torch_mc = TorchMonteCarloExponentialModel(lam1=lam1, lam2=lam2, betas=betas, R_init1=R_init1, R_init2=R_init2,
                                        theta1=theta1, theta2=theta2, N=N, vix_N=0, maturity=maturity,
                                           timestep_per_day=timestep_per_day,
                                           parabolic=parabolic, parabolic_offset=0,
                                          device=device)
torch_mc.simulate(save_R=True)



def model_prices_VR(torch_mc,tensor_strikes_spx,discount_spx,forward_scaled_spx,list_of_mat_idxs_spx,
                    tensor_spx_mat,idxs_spx,device,variance_reduction):
    
    if variance_reduction:
        return torch.stack([torch_mc.compute_option_price(strikes=tensor_strikes_spx[i], option_maturity=tensor_spx_mat[i],
                                                          scaled_fwd=forward_scaled_spx[i], discount=discount_spx[i], return_future=False,
                                 var_reduction=True, sigma0=0.1) for i in idxs_spx])
    else:
        return torch.stack([torch_mc.compute_option_price(strikes=tensor_strikes_spx[i], option_maturity=tensor_spx_mat[i],
                                                          scaled_fwd=forward_scaled_spx[i], discount=discount_spx[i], return_future=False,
                                 var_reduction=False, sigma0=0.1) for i in idxs_spx])

    
############################# DATA SPX & VIX
def get_types_of_elements(input_dict):
    types_list = []

    for key, value in input_dict.items():
        element_type = type(value)
        types_list.append(element_type)

    return types_list

print('Types elements SPX dictionary:\n',get_types_of_elements(loaded_dict_spx))

print('Types elements VIX dictionary:\n',get_types_of_elements(loaded_dict_vix))


    ### SPX
    
x=[int(m.split('.')[0])/scaling_maturities for m in maturities_spx]
indices=np.where(np.isin(loaded_dict_spx['maturities_spx'], x))[0].astype(int)
print(indices)
tensor_spx_mat=convert_to_tensor_arr(np.array([loaded_dict_spx['maturities_spx'][i] for i in indices])).to(device)
discount_spx=convert_to_tensor_arr(np.array([loaded_dict_spx['discount_spx'][i] for i in indices])).to(device)
forward_scaled_spx=convert_to_tensor_arr(np.array([loaded_dict_spx['forward_scaled_spx'][i] for i in indices])).to(device)
tensor_strikes_spx=numpy_arrays_to_tensors([loaded_dict_spx['strikes_spx'][i] for i in indices],device)
tensor_bid_ivol_spx=numpy_arrays_to_tensors([loaded_dict_spx["bid"][i] for i in indices],device)
tensor_ask_ivol_spx=numpy_arrays_to_tensors([loaded_dict_spx["ask"][i] for i in indices],device)
tensor_mid_ivol_spx=numpy_arrays_to_tensors([loaded_dict_spx["mid"][i] for i in indices],device)
tensor_las_vegas_spx=numpy_arrays_to_tensors([el/sum(el) for el in [loaded_dict_spx['las_vegas_by_mat_spx'][i] for i in indices]],device)


### VIX
x=[int(m.split('.')[0])/scaling_maturities for m in maturities_vix]
indices=np.where(np.isin(loaded_dict_vix['maturities_vix'], x))[0].astype(int)

tensor_vix_mat=convert_to_tensor_arr(np.array([loaded_dict_vix['maturities_vix'][i] for i in indices])).to(device)
tensor_vix_futures=convert_to_tensor_arr(np.array([loaded_dict_vix['future'][i] for i in indices])).to(device)
discount_vix=convert_to_tensor_arr(np.array([loaded_dict_vix['discount_vix'][i] for i in indices])).to(device)

tensor_prices_vix=[convert_to_tensor_arr(loaded_dict_vix["prices_vix"][i]).to(device) for i in indices]    
tensor_bid_ivol_vix=[convert_to_tensor_arr(loaded_dict_vix["bid"][i]).to(device) for i in indices]  
tensor_ask_ivol_vix=[convert_to_tensor_arr(loaded_dict_vix["ask"][i]).to(device) for i in indices]  
tensor_mid_ivol_vix=[convert_to_tensor_arr(loaded_dict_vix["mid"][i]).to(device) for i in indices]  
torch_strikes_vix_unnormalized=[convert_to_tensor_arr(loaded_dict_vix["strikes_vix"][i]).to(device) for i in indices]  
tensor_las_vegas_vix=[convert_to_tensor_arr(loaded_dict_vix["las_vegas_by_mat_vix"][i]).to(device)/torch.sum(convert_to_tensor_arr(loaded_dict_vix["las_vegas_by_mat_vix"][i])) for i in indices]  
    

list_of_mat_idxs_spx=[int(torch.ceil(tensor_spx_mat[j] / torch_mc.timestep)) for j in range(len(tensor_spx_mat))]
list_of_mat_idxs_spx=torch.tensor(list_of_mat_idxs_spx)

trial=model_prices_VR(torch_mc,tensor_strikes_spx,discount_spx,forward_scaled_spx,list_of_mat_idxs_spx,tensor_spx_mat,[0,1],device,False)
print('Model prices:\n',trial)

print('Las Vegas SPX:',tensor_las_vegas_spx)


print('Las Vegas VIX:',tensor_las_vegas_vix)

############################# LOAD THE NEURAL NETWORK

directory_pre_trained_network=r'/home/'+user+'/PDV/vix_nested/parameter_set_1/all_nns/'
os.chdir(directory_pre_trained_network)

loaded_model = keras.models.load_model(name_network,compile=False)
########################## DISPLAY ARCHITECTURE
for i, layer in enumerate (loaded_model.layers):
    print (i, layer)
    try:
        print ("    ",layer.activation)
    except AttributeError:
        print('no activation attribute')

os.chdir(directory_pre_trained_network+name_network+'_scaling')
mean_scaling_VIX, std_scaling_VIX = np.load('mean_scaling_VIX.npy',allow_pickle=True),np.load('std_scaling_VIX.npy',allow_pickle=True)
mean_scaling_Rs, std_scaling_Rs= np.load('mean_scaling_Rs.npy',allow_pickle=True),np.load('std_scaling_Rs.npy',allow_pickle=True)
mean_scaling_parameters, std_scaling_parameters= np.load('mean_scaling_parameters.npy',allow_pickle=True),np.load('std_scaling_parameters.npy',allow_pickle=True)



############################### PRICING VIX - EXAMPLE (FOR THE PRICING FUNCTIONS SEE THE MODULE pricing_vix_neural.py)


if flag_run_pricing_examples:
        
    parameters=torch.tensor([35,35,5,1,0.04,-0.12,0.7,0.11,0.6,0.3])
    lam1=parameters[:2]
    lam2=parameters[2:4]
    betas=parameters[4:7]
    parabolic=parameters[7]
    theta1=parameters[-2]
    theta2=parameters[-1] 
    
    R_init1 = initialize_R(lam1,dt=dt, past_prices=spx, transform=identity)
    R_init2 = initialize_R(lam2,dt=dt, past_prices=spx, transform=squared)
    maturity=0.1 #in years
    N=5000
    timestep_per_day = 1 # number of timestep per business day
    torch_mc = TorchMonteCarloExponentialModel(lam1=lam1, lam2=lam2, betas=betas, R_init1=R_init1,
                                               R_init2=R_init2,theta1=theta1, theta2=theta2, N=N, vix_N=4000,
                                               maturity=maturity, parabolic=parabolic, parabolic_offset=torch.tensor(0),device=device)
    
    torch_mc.simulate(save_R=True)
    list_of_mat_idxs_spx=torch.stack([torch_mc.index_of_timestamp(tensor_spx_mat[j]) for j in range(len(tensor_spx_mat))]).to(device)
    list_of_mat_idxs_vix=torch.stack([torch_mc.index_of_timestamp(tensor_vix_mat[j]) for j in range(len(tensor_vix_mat))]).to(device)
    print(tensor_spx_mat*365)
    print(list_of_mat_idxs_spx)
    print(tensor_vix_mat*365)
    print(list_of_mat_idxs_vix)
    
    vixy=neural_vix(loaded_model,parameters,torch_mc,mean_scaling_VIX,std_scaling_VIX,
                    mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,
                    std_scaling_Rs,list_of_mat_idxs_vix,N,[0],device)
    
    print('Model Future:\n',vixy.mean())
    print('Model VIX prices:\n',torch.mean(torch_payoff_call_vix_select_mat(vixy, torch_strikes_vix_unnormalized,discount_vix,tensor_vix_mat,[0],device)[0],axis=0))

########################## END EXAMPLE



# ###################################### DEFINITION OF THE LOSS FUNCTIONS


def soft_indicator(x):
    "x has to be a number (int/float), this function is a smooth approximation of the indicator function."
    U=0.5
    return U*(torch.tanh(x*100)+U*2)


names_params=['lam10','lam11','lam20','lam21','beta0','beta1','beta2','beta12','theta1','theta2']

omega_spx, omega_vix, omega_futures = 20, 10,20



def loss_vix_prices_torch(parameters,torch_strikes_vix_unnormalized,tensor_prices_vix,
                          tensor_vix_futures,tensor_las_vegas_vix,tensor_bid_vix,tensor_ask_vix,
                          loaded_model,torch_mc,mean_scaling_VIX,std_scaling_VIX,
                          mean_scaling_parameters,discount_vix,tensor_vix_mat,
                          std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,
                          list_of_mat_idxs_vix,N,idxs_vix,device,omega_vix,omega_futures,flag_neural):
    
    if flag_neural:
        parameters=convert_to_tensor_arr(parameters).to(device)
        VIX=neural_vix(loaded_model,parameters,torch_mc,mean_scaling_VIX,std_scaling_VIX,mean_scaling_parameters,
                    std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,list_of_mat_idxs_vix,N,idxs_vix,device)
        VIX_futures=torch_futures(VIX).squeeze(1).cpu()
        VIX_payoffs=torch_payoff_call_vix_select_mat(VIX, torch_strikes_vix_unnormalized,discount_vix,tensor_vix_mat,idxs_vix,device)
        print('Neural futures;',VIX_futures)
    else:
        VIX=torch_mc.compute_vix(tensor_vix_mat[0], n_batch=torch_mc.vix_N)[None,:,None]*100
        VIX_futures=torch_futures(VIX).squeeze(1).cpu()
        VIX_payoffs=torch_payoff_call_vix_select_mat(VIX, torch_strikes_vix_unnormalized,discount_vix,tensor_vix_mat,idxs_vix,device)
        print('Nested futures;',VIX_futures)

    tensor_vix_futures=tensor_vix_futures.cpu()   
    diff=0
    over_len_idxs=1/len(idxs_vix)
    for i,idx in enumerate(idxs_vix):
        mc_prices_vix=torch.mean(VIX_payoffs[i],axis=0).cpu()
        nbr_strikes_per_mat=mc_prices_vix.shape[0]
        VIX_future_i=VIX_futures[i]
        diff_1= (1/nbr_strikes_per_mat)*(mc_prices_vix/tensor_prices_vix[idx].cpu()-1)**2
        #diff_1=tensor_las_vegas_vix[idx].cpu()*diff_1  #Uncomment to enforce Vega Weigths # 1/nbr_strikes of vix 
        diff_2=(VIX_future_i/tensor_vix_futures[idx]-1)**2
        diff+=omega_vix*(over_len_idxs)*torch.mean(diff_1)+omega_futures*(over_len_idxs)*torch.mean(diff_2)
    return diff


flag_sum=True


def loss_spx_ivs(torch_mc,tensor_mid_ivol_spx,tensor_bid_ivol_spx,tensor_ask_ivol_spx,tensor_las_vegas_spx,tensor_strikes_spx,
                          tensor_spx_mat,discount_spx,forward_scaled_spx,list_of_mat_idxs_spx,N,idxs_spx,device,omega_spx,flag_var_red,flag_sum=True):

    
    SPX_model_prices=model_prices_VR(torch_mc,tensor_strikes_spx,discount_spx,
                                      forward_scaled_spx,list_of_mat_idxs_spx,
                                      tensor_spx_mat,idxs_spx,device,flag_var_red)
    
    diff=0
    for i,idx in enumerate(idxs_spx):
        nbr_strikes_per_mat=tensor_strikes_spx[idx].shape[0]
        ones=torch.ones(nbr_strikes_per_mat).to(device)
        iv_per_mat=[find_ivol_bnp(SPX_model_prices[i][k].cpu().numpy(), 1, strike.detach().cpu().numpy(), tensor_spx_mat[idx].cpu().numpy(), forward_scaled_spx[idx].cpu().numpy(), discount_spx[idx].cpu().numpy()) for k,strike in enumerate(tensor_strikes_spx[idx])]
        iv_per_mat=torch.as_tensor(iv_per_mat).to(device)
        if flag_sum:
            #softy=soft_indicator(tensor_bid_ivol_spx[idx]-iv_per_mat)+soft_indicator(iv_per_mat-tensor_ask_ivol_spx[idx])
            
            softy=1
            
            if i==0:
                diff_ivs=((iv_per_mat/tensor_mid_ivol_spx[idx]-ones)*softy)**2
            else:
                diff_ivs=((iv_per_mat/tensor_mid_ivol_spx[idx]-ones)*softy)**2
            
            #diff_ivs=(diff_ivs.cpu()*tensor_las_vegas_spx[idx].cpu()) #
            diff+=(1/nbr_strikes_per_mat)*omega_spx*(1/len(idxs_spx))*torch.mean(diff_ivs) #
        else:
            softy=soft_indicator(tensor_bid_ivol_spx[idx]-iv_per_mat)+soft_indicator(iv_per_mat-tensor_ask_ivol_spx[idx])
            softy=1
            diff_ivs=(torch.div(iv_per_mat,tensor_mid_ivol_spx[i])-ones)*softy
            diff_ivs=torch.multiply(diff_ivs.cpu(),tensor_las_vegas_spx[idx].cpu()) 
            diff+=omega_spx*(1/len(idxs_spx))*diff_ivs
    return diff



if flag_joint:
    idxs_spx=[0,1]
    idxs_vix=[0]
    
elif flag_only_spx:
    idxs_spx=[0,1,2,3,4,5,6,7,8,9,11]
    idxs_vix=[0]

if flag_only_monthly:
    dir_calibration_idxs=dir_calibration_monthly+'/SPX_'+str(idxs_spx)+'_VIX_'+str(idxs_vix)
    print(dir_calibration_idxs)
    os.makedirs(dir_calibration_idxs,exist_ok=True)
else:
    dir_calibration_idxs=dir_calibration+'/SPX_'+str(idxs_spx)+'_VIX_'+str(idxs_vix)
    os.makedirs(dir_calibration_idxs,exist_ok=True)
    
    

############################################################################## INITIAL SEARCH

'''
Here several inputs should actually taken into account, namely all the market data above;
We just point out the additional one to specify the sampling of the initial parameters before optimization:
    
nbr_samples_hyper (int): number of configurations to be sampled
timesteps_per_day (int): number of timesteps per day in the discretization of the model (>1 for continuous model)
variance_reduction (boolean): if True enforces the variance reduction for Call options
omega_spx (float): additional weight on SPX loss
omega_vix (float): additional weight on VIX loss
omega_futures (float): additional weight on VIX futures

return df_params_init (pandas object with as column the initial parameters for SPX, VIX or JOINT)

df_params_init.csv is also stored locally with the graphs of the corresponding IV Smiles

'''


if flag_load_initial_guess==False:
    nbr_samples_hyper=10000
    timesteps_per_day=2 
    variance_reduction=False 
    #omega_spx, omega_vix, omega_futures = 20, 10,20
    omega_spx, omega_vix, omega_futures = 30, 10,20

    lam10,lam11,beta1,beta12,theta1=get_samples_lambdas_condition_beta1(1,100,nbr_samples_hyper,1,True)
    lam20,lam21=get_samples_lambdas_condition_beta1(1,100,nbr_samples_hyper,2,True) 
    beta0=gen_rand_torch(0,0.15,1,nbr_samples_hyper).squeeze(0)
    beta2=gen_rand_torch(0,0.9,1,nbr_samples_hyper).squeeze(0) 
    theta2=gen_rand_torch(0,1,1,nbr_samples_hyper).squeeze(0)
    beta12=beta12.squeeze(0)

    R_init1_list=[]
    R_init2_list=[]


    for k in tqdm(range(nbr_samples_hyper),desc='Trying different configurations:'):  
            select_config=k
            lam10_k=lam10[select_config]
            lam11_k=lam11[select_config]
            lam20_k=lam20[select_config]
            lam21_k=lam21[select_config]
            R_init1,R_init2=get_initial_values_Rs_updated(torch.stack([lam10_k,lam11_k]),torch.stack([lam20_k,lam21_k]),spx)
            R_init1_list.append(R_init1)
            R_init2_list.append(R_init2)   
            
    R_init1_list2=torch.stack(R_init1_list).T
    R_init2_list2=torch.stack(R_init2_list).T
    
    losses_SPX=[]
    losses_VIX=[]
    
    if variance_reduction:
        N=100000
    else:
        N=200000
    #N=200000
    timestep_per_day = 2 # number of timestep per business day
    
    for k in tqdm(range(nbr_samples_hyper),desc='Hyperparameter search'):
        lam10_k,lam11_k,beta1_k,beta12_k,theta1_k=lam10[k],lam11[k],beta1[k],beta12[k],theta1[k]
        lam20_k,lam21_k=lam20[k], lam21[k]
        beta0_k=beta0[k]
        beta2_k=beta2[k]
        theta2_k=theta2[k]
        R1_init_k=R_init1_list2[:,k]
        R2_init_k=R_init2_list2[:,k]
        maturity=get_maximal_mat(idxs_vix,idxs_spx,tensor_spx_mat,tensor_vix_mat)+dt
    
        torch_mc1 = TorchMonteCarloExponentialModel(lam1=torch.tensor([lam10_k,lam11_k]), lam2=torch.tensor([lam20_k,lam21_k]),
                                                    betas=torch.tensor([beta0_k, beta1_k, beta2_k]), R_init1=R1_init_k,
                                                    R_init2=R2_init_k,theta1=theta1_k, theta2=theta2_k, N=N, vix_N=0,
                                                    maturity=maturity, parabolic=beta12_k,
                                                    parabolic_offset=torch.tensor(0),timestep_per_day=timestep_per_day,
                                                    fixed_seed=True, seed_root=1,
                                                    device=device)  
        torch_mc1.simulate(save_R=True)
        list_of_mat_idxs_spx=torch.stack([torch_mc1.index_of_timestamp(tensor_spx_mat[j]) for j in range(len(tensor_spx_mat))])
        list_of_mat_idxs_spx.to(device)
        list_of_mat_idxs_vix=torch.stack([torch_mc1.index_of_timestamp(tensor_vix_mat[j]) for j in range(len(tensor_vix_mat))])
        list_of_mat_idxs_vix.to(device)
    
        parameters=torch.stack([lam10_k,lam11_k,lam20_k,lam21_k,beta0_k,beta1_k,beta2_k,beta12_k,theta1_k.squeeze(0),theta2_k.squeeze(0)]).to(device)
        
        LOSS_IV_SPX=loss_spx_ivs(torch_mc1,tensor_mid_ivol_spx,tensor_bid_ivol_spx,
                                  tensor_ask_ivol_spx,tensor_las_vegas_spx,tensor_strikes_spx,
                              tensor_spx_mat,discount_spx,forward_scaled_spx,list_of_mat_idxs_spx,
                              N,idxs_spx,device,omega_spx,variance_reduction,True)
        

        LOSS_PRICES_VIX=loss_vix_prices_torch(parameters,torch_strikes_vix_unnormalized,tensor_prices_vix,
                              tensor_vix_futures,tensor_las_vegas_vix,tensor_bid_ivol_vix,tensor_ask_ivol_vix,
                              loaded_model,torch_mc1,mean_scaling_VIX,std_scaling_VIX,mean_scaling_parameters,
                              discount_vix,tensor_vix_mat,std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,
                              list_of_mat_idxs_vix,N,idxs_vix,device,omega_vix,omega_futures,True)
        
        losses_VIX.append(LOSS_PRICES_VIX)
        losses_SPX.append(LOSS_IV_SPX)
        
        
    list_values=[]
    for value in np.linspace(0,1,100):
        alpha=value
        idx_min_joint_err=torch.argmin(alpha*(1/len(idxs_spx))*torch.tensor(losses_SPX)+(1-alpha)*(1/len(idxs_vix))*torch.tensor(losses_VIX))
        list_values.append(idx_min_joint_err)
    
    
    idx_min_spx_err=torch.argmin(torch.tensor(losses_SPX))
    idx_min_vix_err=torch.argmin(torch.tensor(losses_VIX))
    
    alpha=0.5
    idx_min_joint_err=torch.argmin(alpha*(1/len(idxs_spx))*torch.tensor(losses_SPX)*0.5+(1-alpha)*(1/len(idxs_vix))*torch.tensor(losses_VIX))
    
    dc_indices_hyperparameters={'idx0':idx_min_joint_err,
                                'idx1':idx_min_spx_err,
                                'idx2':idx_min_vix_err}
    
    
    print('IDX0', idx_min_joint_err)
    print('IDX1', idx_min_spx_err)
    print('IDX2', idx_min_vix_err)
    
    
    parameters_init_dict={}
   
    if variance_reduction:
        N=200000
    else:
        N=400000
    
    for u in range(3):
        dir_initial_guess=dir_calibration_idxs+'/initial_guess_'+str(u)
        os.makedirs(dir_initial_guess,exist_ok=True)
        os.chdir(dir_initial_guess)
        
        SELECT_IDX='idx'+str(u)
    
        k=int(dc_indices_hyperparameters[SELECT_IDX])
        lam10_k,lam11_k,theta1_k,theta2_k=lam10[k],lam11[k],theta1[k],theta2[k]
        lam20_k,lam21_k=lam20[k], lam21[k]    
        beta0_k, beta1_k, beta2_k, beta12_k=beta0[k], beta1[k], beta2[k], beta12[k]
        
        R1_init_k=R_init1_list2[:,k]
        R2_init_k=R_init2_list2[:,k]
        maturity=get_maximal_mat(idxs_vix,idxs_spx,tensor_spx_mat,tensor_vix_mat)+dt
    
        torch_mc_ = TorchMonteCarloExponentialModel(lam1=torch.tensor([lam10_k,lam11_k]), lam2=torch.tensor([lam20_k,lam21_k]),
                                                    betas=torch.tensor([beta0_k, beta1_k, beta2_k]), R_init1=R1_init_k,
                                                    R_init2=R2_init_k,theta1=theta1_k, theta2=theta2_k, N=N, vix_N=0,
                                                    maturity=maturity, parabolic=beta12_k,
                                                    timestep_per_day = timestep_per_day,
                                                    fixed_seed=True, seed_root=1,
                                                    parabolic_offset=torch.tensor(0),device=device)
        
        torch_mc_.simulate(save_R=True)
        
        list_of_mat_idxs_spx=torch.stack([torch_mc_.index_of_timestamp(tensor_spx_mat[j]) for j in range(len(tensor_spx_mat))])
        list_of_mat_idxs_spx.to(device)
        
        list_of_mat_idxs_vix=torch.stack([torch_mc_.index_of_timestamp(tensor_vix_mat[j]) for j in range(len(tensor_vix_mat))])
        list_of_mat_idxs_vix.to(device)
    
        parameters_init=torch.stack([lam10_k,lam11_k,lam20_k,lam21_k,beta0_k,beta1_k,beta2_k,beta12_k,theta1_k.squeeze(0),theta2_k.squeeze(0)]).to(device)
    
        if SELECT_IDX[-1]=='0':
            parameters_init_dict['init_JOINT']=parameters_init.cpu().numpy()
        elif SELECT_IDX[-1]=='1':
            parameters_init_dict['init_SPX']=parameters_init.cpu().numpy()
        elif SELECT_IDX[-1]=='2':
            parameters_init_dict['init_VIX']=parameters_init.cpu().numpy()
    
        ##########################################################################################

        
        SPX_model_prices=model_prices_VR(torch_mc_,tensor_strikes_spx,discount_spx,
                                          forward_scaled_spx,list_of_mat_idxs_spx,
                                          tensor_spx_mat,idxs_spx,device,variance_reduction)
        
        
        VIX=neural_vix(loaded_model,parameters_init,torch_mc_,mean_scaling_VIX,std_scaling_VIX,mean_scaling_parameters,
                    std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,list_of_mat_idxs_vix,N,idxs_vix,device)
        VIX_futures=torch_futures(VIX).squeeze(1)
        VIX_payoffs=torch_payoff_call_vix_select_mat(VIX, torch_strikes_vix_unnormalized,discount_vix,tensor_vix_mat,idxs_vix,device)


        for r,idx in enumerate(idxs_vix):
            VIX_prices=torch.mean(VIX_payoffs[r],axis=0)
            iv_init_VIX=[iv_black(VIX_prices[k].cpu().numpy(), strike, tensor_vix_mat[idx].cpu().numpy(), float(VIX_futures.cpu().numpy()[r]), discount_vix[idx].cpu().numpy()) for k,strike in enumerate(torch_strikes_vix_unnormalized[idx].cpu().numpy())]


            plt.figure(figsize=(8, 6))
            plt.plot(torch_strikes_vix_unnormalized[idx].cpu(),VIX_prices.cpu(),marker='o',color='blue',label='Model')
            plt.plot(torch_strikes_vix_unnormalized[idx].cpu(),tensor_prices_vix[idx].cpu(),marker='*',color='red',label='Market')
            plt.axvline(x=VIX_futures[r].cpu(),linestyle='--',color='blue')
            plt.axvline(x=tensor_vix_futures[idx].cpu(),linestyle='--',color='red')
            plt.legend()
            plt.grid()
            plt.title('VIX prices at T='+maturities_vix[idx].split('d')[0].split('.')[0]+' '+maturities_vix[idx][-4:])
            plt.savefig('VIX_prices_'+str([idxs_vix[r]])+'.png',dpi=150)
            plt.close()
            plt.show()
    
            plt.figure(figsize=(8, 6))
            plt.plot(torch_strikes_vix_unnormalized[idx].cpu(),np.array(iv_init_VIX),label='Model',marker='o',alpha=0.5,color='blue',)
            plt.scatter(torch_strikes_vix_unnormalized[idx].cpu(),tensor_bid_ivol_vix[idx].cpu(),marker='*',color='red')
            plt.scatter(torch_strikes_vix_unnormalized[idx].cpu(),tensor_ask_ivol_vix[idx].cpu(),marker='*',color='red')
            plt.fill_between(torch_strikes_vix_unnormalized[idx].cpu().numpy(),tensor_bid_ivol_vix[idx].cpu().numpy(),tensor_ask_ivol_vix[idx].cpu().numpy(),color='red',alpha=0.3,label='Bid/Ask')
            plt.axvline(x=VIX_futures[r].cpu(),linestyle='--',color='blue')
            plt.axvline(x=tensor_vix_futures[idx].cpu(),linestyle='-.',color='red')
            plt.legend()
            plt.grid()
            plt.title('Initial IV T= '+maturities_vix[idx].split('d')[0].split('.')[0]+' '+maturities_vix[idx][-4:])
            plt.savefig('VIX_IV_'+str([idxs_vix[r]])+'.png',dpi=150)
            plt.close()
            plt.show()

            
        ivs_spx_initial=[]
        for j,idx in enumerate(idxs_spx):
            ivs_spx_initial.append([find_ivol_bnp(SPX_model_prices[j][k].cpu().numpy(), 1, strike.numpy(), tensor_spx_mat[idx].cpu().numpy(), forward_scaled_spx[idx].cpu().numpy(), discount_spx[idx].cpu().numpy()) for k,strike in enumerate(tensor_strikes_spx[idx].cpu())])
    
    
        for j,idx in enumerate(idxs_spx):       
            plt.figure(figsize=(8, 6))
            plt.plot(tensor_strikes_spx[idx].cpu(),np.array(ivs_spx_initial[j]),label='Model',marker='o',alpha=0.5,color='blue',)
            plt.scatter(tensor_strikes_spx[idx].cpu(),tensor_bid_ivol_spx[idx].cpu(),marker='*',color='red')
            plt.scatter(tensor_strikes_spx[idx].cpu(),tensor_ask_ivol_spx[idx].cpu(),marker='*',color='red')
            plt.fill_between(tensor_strikes_spx[idx].cpu().numpy(),tensor_bid_ivol_spx[idx].cpu().numpy(),tensor_ask_ivol_spx[idx].cpu().numpy(),color='red',alpha=0.3,label='Bid/Ask')
            plt.legend()
            plt.grid()
            plt.title('Initial IV T= '+maturities_spx[idx].split('d')[0].split('.')[0]+' '+maturities_spx[idx][-4:])
            plt.savefig('SPX_IV_'+str([idxs_spx[j]])+'.png',dpi=150)
            plt.close()
            plt.show()
    df_params_init=pd.DataFrame(parameters_init_dict,index=names_params)
    os.chdir(dir_calibration_idxs)
    df_params_init.to_csv('parameters_init.csv')  


else:
    os.chdir(dir_calibration_idxs)
    #df_params_init=pd.read_csv('parameters_init.csv')
    


# =============================================================================
# 
# 
#  LOSS FUNCTION FOR JOINT CALIBRATION
# 
# 
# =============================================================================

def loss_joint(parameters,spx,idxs_spx,idxs_vix,N,C,omega_vix,
                omega_spx, omega_futures,timestep_per_day,flag_var_red,flag_neural):
    
    '''
    TO DO 
    
    '''
    
    
    
    
    
    parameters=torch.tensor(parameters).clone().detach().to(device)
    lam1=parameters[:2]
    lam2=parameters[2:4]
    betas=parameters[4:7]
    parabolic=parameters[7]
    theta1=parameters[-2]
    theta2=parameters[-1] 
    
    R_init1 = initialize_R(lam1,dt=dt, past_prices=spx, transform=identity)
    R_init2 = initialize_R(lam2,dt=dt, past_prices=spx, transform=squared)
    
    
    if lam1[0]>lam1[1]:
        pass
    else:
        theta1=1-theta1
        lam1=torch.flip(lam1,(0,))
        R_init1=torch.flip(R_init1,(0,))
        
    if lam2[0]>lam2[1]:
        pass
    else:
        theta2=1-theta2
        lam2=torch.flip(lam2,(0,))
        R_init2=torch.flip(R_init2,(0,))
        
    
    maturity=get_maximal_mat(idxs_vix,idxs_spx,tensor_spx_mat,tensor_vix_mat)+dt
    
    torch_mc2_ = TorchMonteCarloExponentialModel(lam1=lam1, lam2=lam2, betas=betas, R_init1=R_init1,
                                                R_init2=R_init2,theta1=theta1, theta2=theta2, N=N, vix_N=4000,
                                                maturity=maturity, parabolic=parabolic,
                                                  parabolic_offset=torch.tensor(0),
                                                  timestep_per_day=timestep_per_day,
                                                  device=device,fixed_seed=True, seed_root=1)
    if C!=1: 
        torch_mc2_.simulate(save_R=True) # If we have to price VIX options than we need the values of the factors at maturity
        list_of_mat_idxs_spx=torch.stack([torch_mc2_.index_of_timestamp(tensor_spx_mat[j]) for j in range(len(tensor_spx_mat))])
        list_of_mat_idxs_vix=torch.stack([torch_mc2_.index_of_timestamp(tensor_vix_mat[j]) for j in range(len(tensor_vix_mat))])
        list_of_mat_idxs_spx.to(device)
        list_of_mat_idxs_vix.to(device)
    else:
        torch_mc2_.simulate(save_R=False) # If we have to price only SPX options than we do not need the values of the factors at maturity
        list_of_mat_idxs_spx=torch.stack([torch_mc2_.index_of_timestamp(tensor_spx_mat[j]) for j in range(len(tensor_spx_mat))])
        list_of_mat_idxs_spx.to(device)

    if C==0:
        LOSS_VIX=loss_vix_prices_torch(parameters,torch_strikes_vix_unnormalized,tensor_prices_vix,
                          tensor_vix_futures,tensor_las_vegas_vix,tensor_bid_ivol_vix,tensor_ask_ivol_vix,
                          loaded_model,torch_mc2_,mean_scaling_VIX,std_scaling_VIX,mean_scaling_parameters,discount_vix,tensor_vix_mat,
                          std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,
                          list_of_mat_idxs_vix,N,idxs_vix,device,omega_vix,omega_futures,flag_neural)
    
        return LOSS_VIX
    elif C==1:
        
        LOSS_SPX=loss_spx_ivs(torch_mc2_,tensor_mid_ivol_spx,tensor_bid_ivol_spx,
                                  tensor_ask_ivol_spx,tensor_las_vegas_spx,tensor_strikes_spx,
                              tensor_spx_mat,discount_spx,forward_scaled_spx,list_of_mat_idxs_spx,
                              N,idxs_spx,device,omega_spx,variance_reduction,True)

        return LOSS_SPX
    else:
        LOSS_SPX=loss_spx_ivs(torch_mc2_,tensor_mid_ivol_spx,tensor_bid_ivol_spx,
                                  tensor_ask_ivol_spx,tensor_las_vegas_spx,tensor_strikes_spx,
                              tensor_spx_mat,discount_spx,forward_scaled_spx,list_of_mat_idxs_spx,
                              N,idxs_spx,device,omega_spx,variance_reduction,True)
        
        
        LOSS_VIX=loss_vix_prices_torch(parameters,torch_strikes_vix_unnormalized,tensor_prices_vix,
                          tensor_vix_futures,tensor_las_vegas_vix,tensor_bid_ivol_vix,tensor_ask_ivol_vix,
                          loaded_model,torch_mc2_,mean_scaling_VIX,std_scaling_VIX,mean_scaling_parameters,discount_vix,tensor_vix_mat,
                          std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,
                          list_of_mat_idxs_vix,N,idxs_vix,device,omega_vix,omega_futures,flag_neural)
        return (1-C)*LOSS_VIX+C*LOSS_SPX
    



ub=np.array([100.0,100.0,100.0,100.0,0.2,-0.001,0.99,0.3,1.0,1.0])
lb=np.array([1.0,1.0,1.0,1.0,0.0,-0.25,0.01,0,0.0,0.0])

minimum_diff_bounds=min(np.abs(ub-lb))/2

#bounds=((1.0,100.0),(1.0,100.0),(1.0,100.0),(1.0,100.0),(0.0,0.2),(-0.2,-0.001),(0.01,0.99),(0.0,0.3),(0.0,1.0),(0.0,1.0))

bounds=((1.0,110.0),(1.0,110.0),(1.0,110.0),(1.0,110.0),(0.0,0.2),(-0.25,-0.001),(0.01,0.99),(0.0,0.15),(0.0,1.0),(0.0,1.0))
#bounds=((1.0,100.0),(1.0,100.0),(1.0,100.0),(1.0,100.0),(0.0,0.2),(-0.2,-0.001),(0.01,0.99),(0.0,0.09),(0.0,1.0),(0.0,1.0))
list_save_results={} #dictionary where to save the results of the calibrations


def wrapped_function_SPX(parameters):
    N,C, timestep_per_day=100000, 1, 2
    omega_spx=1
    omega_vix=0
    omega_futures=0
    return loss_joint(parameters,spx,idxs_spx,idxs_vix,N,C,omega_vix, omega_spx, omega_futures,timestep_per_day,False,True)

def wrapped_function_JOINT(parameters):
    N, timestep_per_day=100000, 10
    #omega_spx, omega_vix, omega_futures = 30, 10,20
    omega_spx, omega_vix, omega_futures,C =10,2,20,0.5#120, 90,60,0.5 #10,3,20,0.5  #
  
    # parameters[0]=np.max(parameters[:1])
    # parameters[1]=np.min(parameters[:1])
    # parameters[2]=np.max(parameters[2:4])
    # parameters[3]=np.min(parameters[2:4])
   
    return loss_joint(parameters,spx,idxs_spx,idxs_vix,N,C,omega_vix, omega_spx, omega_futures,timestep_per_day,False,True)

def wrapped_function_VIX(parameters):
    N,C, timestep_per_day=100000, 0, 2
    omega_futures=25
    omega_vix=25
    return loss_joint(parameters,spx,idxs_spx,idxs_vix,N,C,omega_vix,omega_spx, omega_futures,timestep_per_day,False,True)



if OPTIMIZER=='PYBOBYQA':
    if flag_only_vix:
        parameters_init=np.array(df_params_init["init_VIX"])
        
        list_save_results_vix={}
        t0 = time.time()
        res_vix=pybobyqa.solve(wrapped_function_VIX,parameters_init,
                                  rhobeg=minimum_diff_bounds,rhoend=1e-9,bounds=(lb,ub),
                                objfun_has_noise=False, print_progress=True,maxfun=2500,
                                seek_global_minimum=True,scaling_within_bounds=True)  
        t1 = time.time()
        list_save_results_vix[OPTIMIZER+'_VIX_'+str(idxs_vix)]=[res_vix.x,res_vix.f,(t1-t0)/60]
    
    
        print('Results:',[res_vix.x,res_vix.f,(t1-t0)/60])
        N,timestep_per_day=300000,2
        
        
        ivs_vix=return_vix_results(list_save_results_vix[OPTIMIZER+'_VIX_'+str(idxs_vix)][0],idxs_vix,idxs_spx,N,timestep_per_day,spx,
                                tensor_spx_mat,tensor_vix_mat,torch_strikes_vix_unnormalized,
                                tensor_bid_ivol_vix,tensor_ask_ivol_vix,tensor_vix_futures,
                                rate_vix,loaded_model,mean_scaling_VIX,std_scaling_VIX,
                                mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,
                                device,maturities_vix,user,day,True,True,flag_only_monthly,OPTIMIZER)
    elif flag_only_spx:
        os.chdir(r'/home/ag_cu-student/PDV/bnp/calibration/20231025/only_monthly/SPX_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]_VIX_[0]/after_calib')
        with open('dictionary_results_SPX_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]_PYBOBYQA.pkl', 'rb') as f:
            aux = pickle.load(f)
        print('Successfully past calib')

        parameters_init=np.array([36.7037,12.180,97.4724,1.3172,0.0365952,-0.219749,0.566652,0.25130,0.466686,0.682552])
        
        list_save_results_spx={}
        t0 = time.time()
        res_spx=pybobyqa.solve(wrapped_function_SPX,parameters_init,#wrapped_function_SPX
                                  rhobeg=minimum_diff_bounds,rhoend=1e-11,bounds=(lb,ub),
                                objfun_has_noise=True, print_progress=True,maxfun=600,
                                seek_global_minimum=True,scaling_within_bounds=True)  
        t1 = time.time()
        list_save_results_spx[OPTIMIZER+'_SPX_'+str(idxs_spx)]=[res_spx.x,res_spx.f,(t1-t0)/60]
        
        print('Results:',[res_spx.x,res_spx.f,(t1-t0)/60])
              
        N,timestep_per_day=400000,1
       
        ivs_spx, iv_calibrated_VIX=return_joint_results(list_save_results_spx['PYBOBYQA_SPX_'+str(idxs_spx)][0],idxs_vix,idxs_spx,N,timestep_per_day,spx,
                                                        tensor_spx_mat,tensor_vix_mat,
                                                        tensor_strikes_spx,tensor_bid_ivol_spx,
                                                        tensor_ask_ivol_spx,
                                                        discount_spx,forward_scaled_spx,torch_strikes_vix_unnormalized,
                                                        tensor_bid_ivol_vix,tensor_ask_ivol_vix,tensor_vix_futures,
                                                        discount_vix,loaded_model,mean_scaling_VIX,std_scaling_VIX,
                                                        mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,
                                                        device,maturities_vix,maturities_spx,user,day,True,True,flag_only_monthly,OPTIMIZER)
        
        

    elif flag_joint:
        
        
        # os.chdir(r'/home/ag_cu-student/PDV/bnp/calibration/20210602/only_monthly/SPX_[0, 1]_VIX_[0]/after_calib')
        # with open('dictionary_results_SPX_[0, 1]_VIX[0]_PYBOBYQA.pkl', 'rb') as f:
        #     aux = pickle.load(f)
        # print('Successfully past calib')
        # parameters_init=aux['parameters']
        
        
        #parameters_init=np.array([6.9385590e+01,7.5323471e+01,4.1326976e+00,1.0912706e+01,2.3609264e-02,-1.3063693e-01,6.7408067e-01,1.0808440e-01,1.1172786e-02,2.5073594e-01])
        
        

        # parameters_init=np.array([ 4.1464874e+01,  4.2581776e+01,  2.9571733e+00,  7.6059980e+00,
        #         3.3099495e-02, -1.7183690e-01,  5.9811962e-01,  1.6826859e-01,
        #         3.1168380e-01,  2.2865449e-01])
                
        #parameters_init=np.array([4.2068699e+01, 3.1237467e+01,3.5785184e+00,2.6375737e+00,
        #                          2.9032653e-02,-1.6604531e-01,6.3238263e-01, 1.7324816e-01,0.29,0.297])
        
        ## 0602
        #parameters_init=np.array([4.5103043e+01, 3.2198799e+01,  4.4956, 2.9505, 2.5286727e-02, -1.6608621e-01,  6.4730632e-01,  1.6383716e-01,  0.2935365438,  0.730883])
        # parameters_init=np.array([4.4998451e+01,4.4536526e+01,5.5069561e+00,2.3918114e+00,
        #                           2.4987632e-02, -1.6493885e-01, 6.0191417e-01, 1.5795968e-01, 
        #                           1.9580939e-01, 1.0-2.1228626e-01])
        
        parameters_init=np.array([ 4.4423229e+01, 3.3194923e+01,4.3104205e+00,3.2544370e+00,2.0539796e-02,-1.6018687e-01,6.9221014e-01,1.6387092e-01, 3.9801088e-01,7.2006655e-01])
        
       #parameters_init=np.array([ 45.0746, 31.4953,3.6164, 3.6162, 0.0237,-0.1601,0.6847,0.16401360929012299, 0.37885382771492004,0.7097825407981873])
        
        #parameters_init=np.array([4.6779125e+01,3.0961702e+01,3.5337327e+00,3.5336852e+00,2.2383163e-02,-1.5914321e-01,6.9871628e-01,1.3276958e-01,3.8170373e-01,7.0905536e-01])

        #parameters_init=np.array([ 4.4631840e+01, 3.1742641e+01,2.8133910e+00,4.0419154e+00,2.4916638e-02,-1.6867863e-01, 6.3328868e-01,1.6464664e-01, 2.6367915e-01,2.0342033e-01])

        print('Init parameters',parameters_init)
        
        list_save_results_joint={}
        t0 = time.time()
        res_joint=pybobyqa.solve(wrapped_function_JOINT,parameters_init,
                                  rhobeg=minimum_diff_bounds,rhoend=1e-12,bounds=(lb,ub),
                                objfun_has_noise=False, print_progress=True,maxfun=1000,
                                seek_global_minimum=True,scaling_within_bounds=True)  
        t1 = time.time()
        list_save_results_joint[OPTIMIZER+'_SPX_'+str(idxs_spx)+'_VIX_'+str(idxs_vix)]=[res_joint.x,res_joint.f,(t1-t0)/60]
           
        print('Results:',[res_joint.x,res_joint.f,(t1-t0)/60])
        
        
        N,timestep_per_day=200000,10
       
        ivs_spx, iv_calibrated_VIX=return_joint_results(list_save_results_joint['PYBOBYQA_SPX_'+str(idxs_spx)+'_VIX_'+str(idxs_vix)][0],idxs_vix,idxs_spx,N,timestep_per_day,spx,
                                                        tensor_spx_mat,tensor_vix_mat,
                                                        tensor_strikes_spx,tensor_bid_ivol_spx,
                                                        tensor_ask_ivol_spx,
                                                        discount_spx,forward_scaled_spx,torch_strikes_vix_unnormalized,
                                                        tensor_bid_ivol_vix,tensor_ask_ivol_vix,tensor_vix_futures,
                                                        discount_vix,loaded_model,mean_scaling_VIX,std_scaling_VIX,
                                                        mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,
                                                        device,maturities_vix,maturities_spx,user,day,True,True,flag_only_monthly,OPTIMIZER,True)
