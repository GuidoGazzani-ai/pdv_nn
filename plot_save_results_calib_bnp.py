# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:35:59 2023

@author: Guido Gazzani
"""

from sampler_PDV_bnp import *
from pricing_vix_neural_bnp import *
import pickle
import matplotlib.pyplot as plt
import os

user='ag_cu-student' 
#Forget the previous user if you are using these functions locally: the user variable is only meant for directories (see below)


def initialize_R(lam, dt,past_prices=None, max_delta=1000, transform=identity):
    """
    Initialize the R_j for the 4FMPDV model.
    :param lam: torch.tensor of size 2. contains \lambda_{j,1} and \lambda_{j,2}
    :param past_prices: pd.Series. Contains the past prices of the asset.
    :param max_delta: int. Number of past returns to use to compute the weighted averages.
    :param transform: should be identity for R_1 and squared for R_2.
    :return: torch.tensor of size 2. containing R_j1 and R_j2
    """
    returns = 1 - past_prices.shift(1) / past_prices
    returns = torch.tensor(returns.values, device=lam.device)[-max_delta:].flip(0)
    timestamps = torch.arange(returns.shape[0], device=lam.device) * dt
    timestamps.unsqueeze_(0)
    lam = lam.unsqueeze(1)
    weights = exp_kernel_GPU(timestamps, lam)
    x = transform(returns)
    return torch.sum(x * weights, dim=1)




def save_dict_to_txt(dictionary, file_path):
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f'{key}: {value}\n')


def return_vix_results(parameters,idxs_vix,idxs_spx,N,timestep_per_day,spx,
                       tensor_spx_mat,tensor_vix_mat,torch_strikes_vix_unnormalized,
                       tensor_bid_ivol_vix,tensor_ask_ivol_vix,tensor_vix_futures,
                       discount_vix,loaded_model,mean_scaling_VIX,std_scaling_VIX,
                       mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,
                       device,maturities_vix,user,day,flag_save_plot,flag_save_data,flag_only_monthly,optimizer):
    dt = 1/252
    parameters=convert_to_tensor_arr(parameters).to(device)
    lam1=parameters[:2]
    lam2=parameters[2:4]
    betas=parameters[4:7]
    parabolic=parameters[7]
    theta1=parameters[-2]
    theta2=parameters[-1]

    R_init1 = initialize_R(lam1,dt=dt, past_prices=spx, transform=identity)
    R_init2 = initialize_R(lam2,dt=dt, past_prices=spx, transform=squared)
    maturity=get_maximal_mat(idxs_vix,idxs_spx,tensor_spx_mat,tensor_vix_mat)+dt

    torch_mc = TorchMonteCarloExponentialModel(lam1=lam1, lam2=lam2, betas=betas, R_init1=R_init1,
                                               R_init2=R_init2,theta1=theta1, theta2=theta2, N=N, vix_N=0,
                                               maturity=maturity, parabolic=parabolic, parabolic_offset=torch.tensor(0),
                                               timestep_per_day=timestep_per_day,fixed_seed=True,
                                               device=device)
    torch_mc.simulate(save_R=True)
    
    list_of_mat_idxs_vix=torch.stack([torch_mc.index_of_timestamp(tensor_vix_mat[j]) for j in range(len(tensor_vix_mat))])
    list_of_mat_idxs_vix.to(device)
    
    VIX=neural_vix(loaded_model,parameters,torch_mc,mean_scaling_VIX,std_scaling_VIX,
                   mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,
                   std_scaling_Rs,list_of_mat_idxs_vix,N,idxs_vix,device)    
    VIX_futures=torch_futures(VIX).squeeze(1)
    VIX_payoffs=torch_payoff_call_vix_select_mat(VIX, torch_strikes_vix_unnormalized,discount_vix,tensor_vix_mat,idxs_vix,device)
    
    if flag_only_monthly:
        dir_calibration=r'/home/'+user+'/PDV/bnp/calibration'+day+'/only_monthly'
    else:
        dir_calibration=r'/home/'+user+'/PDV/bnp/calibration'+day
    dir_calibration_idxs=dir_calibration+'/SPX_'+str(idxs_spx)+'_VIX_'+str(idxs_vix)+'/after_calib'
    os.makedirs(dir_calibration,exist_ok=True)
    os.makedirs(dir_calibration_idxs,exist_ok=True)
    os.chdir(dir_calibration_idxs)
    
    if idxs_vix==[0]:
        iv_calibrated_VIX=[iv_black(VIX_payoffs[0].mean(axis=0)[k].cpu().numpy(), strike, tensor_vix_mat[0].cpu().numpy(), VIX_futures[0].cpu().numpy(), discount_vix[0].cpu().numpy()) for k,strike in enumerate(torch_strikes_vix_unnormalized[0].cpu().numpy())]
        iv_calibrated_VIX=np.array(iv_calibrated_VIX)
        idx=0
        plt.figure(figsize=(8, 6))
        plt.plot(torch_strikes_vix_unnormalized[idx].cpu(),iv_calibrated_VIX,label='Model',marker='o',alpha=0.5,color='blue')
        plt.scatter(torch_strikes_vix_unnormalized[idx].cpu(),tensor_bid_ivol_vix[idx].cpu(),marker='*',color='red')
        plt.scatter(torch_strikes_vix_unnormalized[idx].cpu(),tensor_ask_ivol_vix[idx].cpu(),marker='*',color='red')
        plt.axvline(x=VIX_futures[0].cpu(),linestyle='--',color='blue')
        plt.axvline(x=tensor_vix_futures[idx].cpu(),linestyle='-.',color='red')
        plt.fill_between(torch_strikes_vix_unnormalized[idx].cpu().numpy(),tensor_bid_ivol_vix[idx].cpu().numpy(),tensor_ask_ivol_vix[idx].cpu().numpy(),color='red',alpha=0.3,label='Bid/Ask')
        plt.legend()
        plt.title('Maturity T= '+maturities_vix[idx].split('d')[0].split('.')[0]+' '+maturities_vix[idx][-4:])
        plt.grid()
       # plt.title('Maturity T= '+maturities_vix[idx].split('0')[0][:-1]+' '+maturities_vix[idx].split('0')[1])
        if flag_save_plot:
            plt.savefig('VIX_IV_'+str([idxs_vix[idx]])+'.png',dpi=200)
        plt.close()
        plt.show()
    else:
        iv_calibrated_VIX=[]
        for r,idx in enumerate(idxs_vix):
            VIX_prices=torch.mean(VIX_payoffs[r],axis=0)
            iv_VIX=[iv_black(VIX_prices[k].cpu().numpy(), strike, tensor_vix_mat[idx].cpu().numpy(), float(VIX_futures.cpu().numpy()[r]), discount_vix[idx].cpu().numpy()) for k,strike in enumerate(torch_strikes_vix_unnormalized[idx].cpu().numpy())]
            iv_calibrated_VIX.append(np.array(iv_VIX))
            plt.figure(figsize=(8, 6))
            plt.plot(torch_strikes_vix_unnormalized[idx].cpu(),np.array(iv_VIX),label='Model',marker='o',alpha=0.5,color='blue',)
            plt.scatter(torch_strikes_vix_unnormalized[idx].cpu(),tensor_bid_ivol_vix[idx].cpu(),marker='*',color='red')
            plt.scatter(torch_strikes_vix_unnormalized[idx].cpu(),tensor_ask_ivol_vix[idx].cpu(),marker='*',color='red')
            plt.fill_between(torch_strikes_vix_unnormalized[idx].cpu().numpy(),tensor_bid_ivol_vix[idx].cpu().numpy(),tensor_ask_ivol_vix[idx].cpu().numpy(),color='red',alpha=0.3,label='Bid/Ask')
            plt.axvline(x=VIX_futures[r].cpu(),linestyle='--',color='blue')
            plt.axvline(x=tensor_vix_futures[idx].cpu(),linestyle='-.',color='red')
            plt.legend()
            plt.title('Maturity T= '+maturities_vix[idx].split('d')[0].split('.')[0]+' '+maturities_vix[idx][-4:])
            plt.grid()
            #plt.title('Maturity T= '+maturities_vix[idx].split('0')[0][:-1]+' '+maturities_vix[idx].split('0')[1])
            if flag_save_plot:
                plt.savefig('VIX_IV_'+str([idxs_vix[r]])+'_'+optimizer+'.png',dpi=200)
            plt.close()
            plt.show()

    if flag_save_data:
        dict_={}
        dict_['idxs']=idxs_vix
        dict_['parameters']=parameters.cpu().numpy()
        dict_['iv_calibrated']=iv_calibrated_VIX
        dict_['futures_calibrated']=VIX_futures.cpu().numpy()
        dict_['prices_calibrated']=[VIX_payoffs[r].mean(axis=0).cpu().numpy() for r in range(len(idxs_vix))]
        dict_['strikes']=[torch_strikes_vix_unnormalized[idx].cpu().numpy() for idx in idxs_vix]
        dict_['R_init1']=R_init1.cpu().numpy()
        dict_['R_init2']=R_init2.cpu().numpy()
        dict_['maturity']=tensor_vix_mat[idxs_vix].cpu().numpy()
        
        os.chdir(dir_calibration_idxs)
        with open('dictionary_results_VIX_'+str(idxs_vix)+'_'+optimizer+'.pkl', 'wb') as f:
            pickle.dump(dict_, f)
        save_dict_to_txt(dict_, 'results_VIX_'+str(idxs_vix)+'_'+optimizer+'.txt')
    
    
    return iv_calibrated_VIX


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



def return_spx_results(parameters,idxs_vix,idxs_spx,N,timestep_per_day,spx, 
                       tensor_spx_mat,tensor_vix_mat,tensor_strikes_spx,tensor_bid_ivol_spx,
                       tensor_ask_ivol_spx, maturities_spx,
                       discount_spx,forward_scaled_spx,device,user,day,flag_save_plot,
                       flag_save_data,flag_only_monthly,optimizer,discrete,dict_noises_set,noise_name):
    
    dt = 1/252
    
    if discrete:
        
        lam1=parameters[:2]
        lam2=parameters[2:4]
        betas=parameters[4:7]
        parabolic=parameters[7]
        theta1=parameters[-2]
        theta2=parameters[-1] 
        

        
        
        R_init1 = initialize_R(lam1,dt=dt, past_prices=spx, transform=identity)
        R_init2 = initialize_R(lam2,dt=dt, past_prices=spx, transform=squared)
        
        maturity=get_maximal_mat(idxs_vix,idxs_spx,tensor_spx_mat,tensor_vix_mat)+dt
        
        S_array,vol_array=sampler_Discrete_PDV_Ndim(parameters,R_init1,R_init2,1,maturity,N,dict_noises_set,noise_name,steps_daily=timestep_per_day)
        list_of_mat_idxs_spx=np.array([index_of_timestamp(tensor_spx_mat[j],timestep_per_day) for j in range(len(tensor_spx_mat))])
        
       # from calibration_discrete import model_prices_VR
        
        SPX_model_prices=model_prices_VR(S_array,vol_array,timestep_per_day,tensor_strikes_spx,discount_spx,forward_scaled_spx,list_of_mat_idxs_spx,
                            tensor_spx_mat,idxs_spx,device,False)
        
    else: 
        parameters=convert_to_tensor_arr(parameters).to(device)
        lam1=parameters[:2]
        lam2=parameters[2:4]
        betas=parameters[4:7]
        parabolic=parameters[7]
        theta1=parameters[-2]
        theta2=parameters[-1]
        
        R_init1 = initialize_R(lam1,dt=dt, past_prices=spx, transform=identity)
        R_init2 = initialize_R(lam2,dt=dt, past_prices=spx, transform=squared)
        maturity=get_maximal_mat(idxs_vix,idxs_spx,tensor_spx_mat,tensor_vix_mat)+dt
        
        torch_mc = TorchMonteCarloExponentialModel(lam1=lam1, lam2=lam2, betas=betas, R_init1=R_init1,
                                                   R_init2=R_init2,theta1=theta1, theta2=theta2, N=N, vix_N=0,
                                                   maturity=maturity, parabolic=parabolic, parabolic_offset=torch.tensor(0),
                                                   timestep_per_day=timestep_per_day,fixed_seed=True,
                                                   device=device)
        torch_mc.simulate(save_R=False)
        list_of_mat_idxs_spx=torch.stack([torch_mc.index_of_timestamp(tensor_spx_mat[j]) for j in range(len(tensor_spx_mat))])
        list_of_mat_idxs_spx.to(device)
        
        SPX_model_prices=model_prices_VR(torch_mc,tensor_strikes_spx,discount_spx,forward_scaled_spx,list_of_mat_idxs_spx,tensor_spx_mat,idxs_spx,device,False).cpu()
    
    ivs_spx=[]
    
    if discrete:
        for j,idx in enumerate(idxs_spx):
            ivs_spx_j=[find_ivol_bnp(SPX_model_prices[j][k], 1, strike, tensor_spx_mat[idx], forward_scaled_spx[idx], discount_spx[idx]) for k,strike in enumerate(tensor_strikes_spx[idx])]
            ivs_spx.append(np.array(ivs_spx_j))
        ivs_spx=np.array(ivs_spx)
        
    else:
        
        for j,idx in enumerate(idxs_spx):
            ivs_spx_j=[find_ivol_bnp(SPX_model_prices[j][k].numpy(), 1, strike.cpu().numpy(), tensor_spx_mat[idx].cpu().numpy(), forward_scaled_spx[idx].cpu().numpy(), discount_spx[idx].cpu().numpy()) for k,strike in enumerate(tensor_strikes_spx[idx].cpu())]
            ivs_spx.append(np.array(ivs_spx_j))
        ivs_spx=np.array(ivs_spx)
    
    if flag_only_monthly:
        dir_calibration=r'/home/'+user+'/PDV/bnp/calibration'+day+'/only_monthly'
    else:
        dir_calibration=r'/home/'+user+'/PDV/bnp/calibration'+day
    if discrete:
        dir_calibration_idxs=dir_calibration+'/discrete/SPX_'+str(idxs_spx)+'_VIX_'+str(idxs_vix)+'/after_calib'
    else:
        dir_calibration_idxs=dir_calibration+'/SPX_'+str(idxs_spx)+'_VIX_'+str(idxs_vix)+'/after_calib'
        
    os.makedirs(dir_calibration,exist_ok=True)
    os.makedirs(dir_calibration_idxs,exist_ok=True)
    os.chdir(dir_calibration_idxs)
    
    if discrete:
        for j,idx in enumerate(idxs_spx):       
            plt.figure(figsize=(8, 6))
            plt.plot(tensor_strikes_spx[idx],np.array(ivs_spx[j]),label='Model',marker='o',alpha=0.5,color='blue')
            plt.scatter(tensor_strikes_spx[idx],tensor_bid_ivol_spx[idx],marker='*',color='red')
            plt.scatter(tensor_strikes_spx[idx],tensor_ask_ivol_spx[idx],marker='*',color='red')
            plt.fill_between(tensor_strikes_spx[idx],tensor_bid_ivol_spx[idx],tensor_ask_ivol_spx[idx],color='red',alpha=0.3,label='Bid/Ask')
            plt.legend()
            plt.title('Maturity T= '+maturities_spx[idx].split('d')[0].split('.')[0]+' '+maturities_spx[idx][-4:])
            plt.grid()
         
            if flag_save_plot:
                plt.savefig('SPX_IV_'+str([idxs_spx[j]])+'_'+optimizer+'.png',dpi=200)
                
            plt.close()
            plt.show()
        
    else:
        
    
        for j,idx in enumerate(idxs_spx):       
            plt.figure(figsize=(8, 6))
            plt.plot(tensor_strikes_spx[idx].cpu().numpy(),np.array(ivs_spx[j]),label='Model',marker='o',alpha=0.5,color='blue')
            plt.scatter(tensor_strikes_spx[idx].cpu().numpy(),tensor_bid_ivol_spx[idx].cpu().numpy(),marker='*',color='red')
            plt.scatter(tensor_strikes_spx[idx].cpu().numpy(),tensor_ask_ivol_spx[idx].cpu().numpy(),marker='*',color='red')
            plt.fill_between(tensor_strikes_spx[idx].cpu().numpy(),tensor_bid_ivol_spx[idx].cpu().numpy(),tensor_ask_ivol_spx[idx].cpu().numpy(),color='red',alpha=0.3,label='Bid/Ask')
            plt.legend()
            plt.title('Maturity T= '+maturities_spx[idx].split('d')[0].split('.')[0]+' '+maturities_spx[idx][-4:])
            plt.grid()
         
            if flag_save_plot:
                plt.savefig('SPX_IV_'+str([idxs_spx[j]])+'_'+optimizer+'.png',dpi=200)
                
            plt.close()
            plt.show()
            
    
    
    if flag_save_data:
        
        dict_={}
        dict_['idxs']=idxs_spx
        
        if discrete:
            dict_['parameters']=parameters
            dict_['iv_calibrated']=ivs_spx
            dict_['prices_calibrated']=SPX_model_prices
            dict_['strikes']=tensor_strikes_spx[idxs_spx]
            dict_['R_init1']=R_init1
            dict_['R_init2']=R_init2
            dict_['maturity']=tensor_spx_mat[idxs_spx]
            dict_['noise']=noise_name
            dict_['noise1']=dict_noises_set
        else:
            
            dict_['parameters']=parameters.cpu().numpy()
            dict_['iv_calibrated']=ivs_spx
            dict_['prices_calibrated']=SPX_model_prices.cpu().numpy()
            dict_['strikes']=tensor_strikes_spx[idxs_spx].cpu().numpy()
            dict_['R_init1']=R_init1.cpu().numpy()
            dict_['R_init2']=R_init2.cpu().numpy()
            dict_['maturity']=tensor_spx_mat[idxs_spx].cpu().numpy()
        os.chdir(dir_calibration_idxs)
        with open('dictionary_results_SPX_'+str(idxs_spx)+'_'+optimizer+'.pkl', 'wb') as f:
            pickle.dump(dict_, f)
        save_dict_to_txt(dict_, 'results_SPX_'+str(idxs_spx)+'_'+optimizer+'.txt')

    return ivs_spx



def return_joint_results(parameters,idxs_vix,idxs_spx,N,timestep_per_day,spx, 
                       tensor_spx_mat,tensor_vix_mat,
                       tensor_strikes_spx,tensor_bid_ivol_spx,
                       tensor_ask_ivol_spx,
                       discount_spx,forward_scaled_spx,torch_strikes_vix_unnormalized,
                       tensor_bid_ivol_vix,tensor_ask_ivol_vix,tensor_vix_futures,
                       discount_vix,loaded_model,mean_scaling_VIX,std_scaling_VIX,
                       mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,
                       device,maturities_vix,maturities_spx,user,day,flag_save_plot,flag_save_data,flag_only_monthly,optimizer,flag_neural):
    dt = 1/252
    parameters=convert_to_tensor_arr(parameters).to(device)
    lam1=parameters[:2]
    lam2=parameters[2:4]
    betas=parameters[4:7]
    parabolic=parameters[7]
    theta1=parameters[-2]
    theta2=parameters[-1]

    R_init1 = initialize_R(lam1,dt=dt, past_prices=spx, transform=identity)
    R_init2 = initialize_R(lam2,dt=dt, past_prices=spx, transform=squared)
    maturity=get_maximal_mat(idxs_vix,idxs_spx,tensor_spx_mat,tensor_vix_mat)+dt

    torch_mc = TorchMonteCarloExponentialModel(lam1=lam1, lam2=lam2, betas=betas, R_init1=R_init1,
                                               R_init2=R_init2,theta1=theta1, theta2=theta2, N=N, vix_N=5000,
                                               maturity=maturity, parabolic=parabolic, parabolic_offset=torch.tensor(0),
                                               timestep_per_day=timestep_per_day,fixed_seed=True,
                                               device=device)
    torch_mc.simulate(save_R=True)
    
    list_of_mat_idxs_vix=torch.stack([torch_mc.index_of_timestamp(tensor_vix_mat[j]) for j in range(len(tensor_vix_mat))])
    list_of_mat_idxs_spx=torch.stack([torch_mc.index_of_timestamp(tensor_spx_mat[j]) for j in range(len(tensor_spx_mat))])
    list_of_mat_idxs_vix.to(device)
    list_of_mat_idxs_spx.to(device)
    ## SPX PRICES
    SPX_model_prices=model_prices_VR(torch_mc,tensor_strikes_spx,discount_spx,forward_scaled_spx,list_of_mat_idxs_spx,tensor_spx_mat,idxs_spx,device,False).cpu()
    
    if flag_neural:
        
        ## VIX PAYOFFS
        VIX=neural_vix(loaded_model,parameters,torch_mc,mean_scaling_VIX,std_scaling_VIX,
                       mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,
                       std_scaling_Rs,list_of_mat_idxs_vix,N,idxs_vix,device)    
        VIX_futures=torch_futures(VIX).squeeze(1)
        VIX_payoffs=torch_payoff_call_vix_select_mat(VIX, torch_strikes_vix_unnormalized,discount_vix,tensor_vix_mat,idxs_vix,device)
    else:
        VIX=torch_mc.compute_vix(tensor_vix_mat[0], n_batch=torch_mc.vix_N)[None,:,None]*100
        VIX_futures=torch_futures(VIX).squeeze(1)
        VIX_payoffs=torch_payoff_call_vix_select_mat(VIX, torch_strikes_vix_unnormalized,discount_vix,tensor_vix_mat,idxs_vix,device)

    
    ivs_spx=[]
    for j,idx in enumerate(idxs_spx):
        ivs_spx_j=[find_ivol_bnp(SPX_model_prices[j][k].numpy(), 1, strike.cpu().numpy(), tensor_spx_mat[idx].cpu().numpy(), forward_scaled_spx[idx].cpu().numpy(), discount_spx[idx].cpu().numpy()) for k,strike in enumerate(tensor_strikes_spx[idx].cpu())]
        ivs_spx.append(np.array(ivs_spx_j))
    ivs_spx=np.array(ivs_spx)
    
    if flag_only_monthly:
        dir_calibration=r'/home/'+user+'/PDV/bnp/calibration'+day+'/only_monthly'
    else:
        dir_calibration=r'/home/'+user+'/PDV/bnp/calibration'+day
    dir_calibration_idxs=dir_calibration+'/SPX_'+str(idxs_spx)+'_VIX_'+str(idxs_vix)+'/after_calib'
    os.makedirs(dir_calibration,exist_ok=True)
    os.makedirs(dir_calibration_idxs,exist_ok=True)
    os.chdir(dir_calibration_idxs)
    
    
    
    for j,idx in enumerate(idxs_spx):       
        plt.figure(figsize=(8, 6))
        plt.plot(tensor_strikes_spx[idx].cpu().numpy(),np.array(ivs_spx[j]),label='Model',marker='o',alpha=0.5,color='blue')
        plt.scatter(tensor_strikes_spx[idx].cpu().numpy(),tensor_bid_ivol_spx[idx].cpu().numpy(),marker='*',color='red')
        plt.scatter(tensor_strikes_spx[idx].cpu().numpy(),tensor_ask_ivol_spx[idx].cpu().numpy(),marker='*',color='red')
        plt.fill_between(tensor_strikes_spx[idx].cpu().numpy(),tensor_bid_ivol_spx[idx].cpu().numpy(),tensor_ask_ivol_spx[idx].cpu().numpy(),color='red',alpha=0.3,label='Bid/Ask')
        plt.legend()
        plt.title('Maturity T= '+maturities_spx[idx].split('d')[0].split('.')[0]+' '+maturities_spx[idx][-4:])
        plt.grid()
     
        if flag_save_plot:
            plt.savefig('SPX_IV_'+str([idxs_spx[j]])+'_'+optimizer+'.png',dpi=200)
            
        plt.close()
        plt.show()
   
    if idxs_vix==[0]:
        iv_calibrated_VIX=[iv_black(VIX_payoffs[0].mean(axis=0)[k].cpu().numpy(), strike, tensor_vix_mat[0].cpu().numpy(), VIX_futures[0].cpu().numpy(), discount_vix[0].cpu().numpy()) for k,strike in enumerate(torch_strikes_vix_unnormalized[0].cpu().numpy())]
        iv_calibrated_VIX=np.array(iv_calibrated_VIX)
        idx=0
        plt.figure(figsize=(8, 6))
        plt.plot(torch_strikes_vix_unnormalized[idx].cpu().numpy(),iv_calibrated_VIX,label='Model',marker='o',alpha=0.5,color='blue')
        plt.scatter(torch_strikes_vix_unnormalized[idx].cpu().numpy(),tensor_bid_ivol_vix[idx].cpu().numpy(),marker='*',color='red')
        plt.scatter(torch_strikes_vix_unnormalized[idx].cpu().numpy(),tensor_ask_ivol_vix[idx].cpu().numpy(),marker='*',color='red')
        plt.axvline(x=VIX_futures[0].cpu().numpy(),linestyle='--',color='blue')
        plt.axvline(x=tensor_vix_futures[idx].cpu().numpy(),linestyle='-.',color='red')
        plt.fill_between(torch_strikes_vix_unnormalized[idx].cpu().numpy(),tensor_bid_ivol_vix[idx].cpu().numpy(),tensor_ask_ivol_vix[idx].cpu().numpy(),color='red',alpha=0.3,label='Bid/Ask')
        plt.legend()
        plt.title('Maturity T= '+maturities_vix[idx].split('d')[0].split('.')[0]+' '+maturities_vix[idx][-4:])
        plt.grid()
       # plt.title('Maturity T= '+maturities_vix[idx].split('0')[0][:-1]+' '+maturities_vix[idx].split('0')[1])
        if flag_save_plot:
            plt.savefig('VIX_IV_'+str([idxs_vix[idx]])+'_'+optimizer+'.png',dpi=200)
        plt.close()
        plt.show()
    else:
        iv_calibrated_VIX=[]
        for r,idx in enumerate(idxs_vix):
            VIX_prices=torch.mean(VIX_payoffs[r],axis=0)
            iv_VIX=[iv_black(VIX_prices[k].cpu().numpy(), strike, tensor_vix_mat[idx].cpu().numpy(), float(VIX_futures.cpu().numpy()[r]), discount_vix[idx].cpu().numpy()) for k,strike in enumerate(torch_strikes_vix_unnormalized[idx].cpu().numpy())]
            iv_calibrated_VIX.append(np.array(iv_VIX))
            plt.figure(figsize=(8, 6))
            plt.plot(torch_strikes_vix_unnormalized[idx].cpu().numpy(),np.array(iv_VIX),label='Model',marker='o',alpha=0.5,color='blue',)
            plt.scatter(torch_strikes_vix_unnormalized[idx].cpu().numpy(),tensor_bid_ivol_vix[idx].cpu().numpy(),marker='*',color='red')
            plt.scatter(torch_strikes_vix_unnormalized[idx].cpu().numpy(),tensor_ask_ivol_vix[idx].cpu().numpy(),marker='*',color='red')
            plt.fill_between(torch_strikes_vix_unnormalized[idx].cpu().numpy(),tensor_bid_ivol_vix[idx].cpu().numpy(),tensor_ask_ivol_vix[idx].cpu().numpy(),color='red',alpha=0.3,label='Bid/Ask')
            plt.axvline(x=VIX_futures[r].cpu().numpy(),linestyle='--',color='blue')
            plt.axvline(x=tensor_vix_futures[idx].cpu().numpy(),linestyle='-.',color='red')
            plt.legend()
            plt.title('Maturity T= '+maturities_vix[idx].split('d')[0].split('.')[0]+' '+maturities_vix[idx][-4:])
            plt.grid()
            if flag_save_plot:
                plt.savefig('VIX_IV_'+str([idxs_vix[r]])+'_'+optimizer+'.png',dpi=200)
            plt.close()
            plt.show()
            
            
    if flag_save_data:
        dict_={}
        dict_['idxs_vix']=idxs_vix
        dict_['idxs_spx']=idxs_spx
        dict_['parameters']=parameters.cpu().numpy()
        dict_['iv_calibrated_vix']=iv_calibrated_VIX
        dict_['iv_calibrated_spx']=ivs_spx
        dict_['futures_calibrated']=VIX_futures.cpu().numpy()
        dict_['prices_calibrated_vix']=[VIX_payoffs[r].mean(axis=0).cpu().numpy() for r in range(len(idxs_vix))]
        dict_['prices_calibrated_spx']=SPX_model_prices.numpy()
        dict_['strikes_vix']=[torch_strikes_vix_unnormalized[idx].cpu().numpy() for idx in idxs_vix]
        dict_['strikes_spx']=tensor_strikes_spx[idxs_spx].cpu().numpy()
        dict_['R_init1']=R_init1.cpu().numpy()
        dict_['R_init2']=R_init2.cpu().numpy()
        dict_['maturity_vix']=tensor_vix_mat[idxs_vix].cpu().numpy()
        dict_['maturity_spx']=tensor_spx_mat[idxs_spx].cpu().numpy()
        
        os.chdir(dir_calibration_idxs)
        with open('dictionary_results_SPX_'+str(idxs_spx)+'_VIX'+str(idxs_vix)+'_'+optimizer+'.pkl', 'wb') as f:
            pickle.dump(dict_, f)
            
        save_dict_to_txt(dict_, 'results_SPX_'+str(idxs_spx)+'_VIX'+str(idxs_vix)+'_'+optimizer+'.txt')
            
            
    return ivs_spx, iv_calibrated_VIX

    
    
