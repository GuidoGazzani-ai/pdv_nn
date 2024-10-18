# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:32:28 2023

@author: Guido Gazzani
"""

import torch
from sampler_PDV_bnp import *


def scale_params(mean_scaling_parameters,std_scaling_parameters,parameters,device):
    '''
    mean_scaling_parameters: np.array 1-dim with the mean of training set for the model's parameters
    std_scaling_parameters: np.array 1-dim with the std of training set for the model's parameters
    parameters: torch.tensor 1-dim with the model's parameters (see ordering above), len=10 if beta_12 is included
    
    returns: scaled parameters
    '''
    parameters=parameters.to(device)
    
    mean_scaling_parameters,std_scaling_parameters=convert_to_tensor_arr(mean_scaling_parameters).to(device), convert_to_tensor_arr(std_scaling_parameters).to(device)
    return torch.div(parameters-mean_scaling_parameters,std_scaling_parameters)


def scale_Rs(mean_scaling_Rs,std_scaling_Rs,R,device):
    '''
    mean_scaling_Rs: np.array 1-dim with the mean of training set for the R's values
    std_scaling_Rs: np.array 1-dim with the std of training set for the R's values
    R: torch.tensor of the values of Rs for different samples
    
    returns: scaled R values
    '''
    R=R.to(device)
    
    mean_scaling_Rs,std_scaling_Rs=convert_to_tensor_arr(mean_scaling_Rs).to(device), convert_to_tensor_arr(std_scaling_Rs).to(device)
    return torch.transpose(torch.div(R-mean_scaling_Rs,std_scaling_Rs),0,1)


def scale_VIX(mean_scaling_VIX,std_scaling_VIX,VIX,device):
    '''
    mean_scaling_VIX: np.array 1-dim with the mean of training set for the VIX values
    std_scaling_VIX: np.array 1-dim with the std of training set for the VIX values
    VIX: np.array of the values of VIX for different samples
    
    returns: scaled VIX values
    '''
    VIX=VIX.to(device)
    
    mean_scaling_VIX,std_scaling_VIX=convert_to_tensor_arr(mean_scaling_VIX).to(device), convert_to_tensor_arr(std_scaling_VIX).to(device)
    return ((VIX-mean_scaling_VIX)/std_scaling_VIX).T

def scale_inverse_VIX(mean_scaling_VIX,std_scaling_VIX,VIX,device):
    '''
    Just like function scale_VIX, but applies the inverse scaling
    
    returns: unscaled VIX values
    '''
    mean_scaling_VIX,std_scaling_VIX=convert_to_tensor_arr(mean_scaling_VIX).to(device), convert_to_tensor_arr(std_scaling_VIX).to(device)
    VIX=convert_to_tensor(VIX).to(device)
    return VIX*std_scaling_VIX+mean_scaling_VIX

def R_at_vix_mat_scale_concatenate(torch_mc,mean_scaling_Rs,std_scaling_Rs,list_of_mat_idxs_vix,idxs_vix,device):
    '''
    torch_mc: class of the "Volatility is Mostly Path-Dependent paper"
    mean_scaling_Rs:  np.array 1-dim with the mean of training set for the R's values
    std_scalings_Rs:  np.array 1-dim with the std of training set for the R's values
    list_of_mat_idxs_vix: list of the timestamps for the considered vix maturities
    
    returns. R torch tensor scaled and at VIX maturities
    '''
    R1_array=torch_mc.R1_array.to(device)
    R2_array=torch_mc.R2_array.to(device)       
    R_array=torch.cat((R1_array[list_of_mat_idxs_vix[idxs_vix],:,:],R2_array[list_of_mat_idxs_vix[idxs_vix],:,:]),axis=1)
    R_array_scaled=scale_Rs(mean_scaling_Rs,std_scaling_Rs,R_array,device)
    R_array_scaled=torch.transpose(R_array_scaled,1,2)
    return R_array_scaled

def scale_and_get_X(parameters,torch_mc,mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,list_of_mat_idxs_vix,idxs_vix,device):
    '''
    parameters: torch.tensor 1-dim with the model's parameters (see ordering above), len=10 if beta_12 is included
    mean_scaling_parameters: np.array 1-dim with the mean of training set for the model's parameters
    std_scaling_parameters: np.array 1-dim with the std of training set for the model's parameters
    torch_mc: class of the "Volatility is Mostly Path-Dependent paper"
    mean_scaling_Rs:  np.array 1-dim with the mean of training set for the R's values
    std_scalings_Rs:  np.array 1-dim with the std of training set for the R's values
    
    returns X torch.tensor of dim. 14 (10+4) x nbr_vix_mat x number of samples
    '''
    scaled_R=R_at_vix_mat_scale_concatenate(torch_mc,mean_scaling_Rs,std_scaling_Rs,list_of_mat_idxs_vix,idxs_vix,device)
    augmented_parameters=parameters.unsqueeze(1).repeat(1,scaled_R.shape[1])
    scaled_parameters=scale_params(mean_scaling_parameters,std_scaling_parameters,augmented_parameters,device)
    scaled_parameters=scaled_parameters.unsqueeze(2).repeat(1, 1,scaled_R.shape[-1])
    X=torch.transpose(torch.cat((scaled_parameters,scaled_R),axis=0),1,2)
    return X

def flatten_X(X):
    '''
    Function merges the maturities and the samples for the neural network
    '''
    return X.flatten(start_dim=1,end_dim=2)

def neural_vix(loaded_model,parameters,torch_mc,mean_scaling_VIX,std_scaling_VIX,
               mean_scaling_parameters,std_scaling_parameters,mean_scaling_Rs,
               std_scaling_Rs,list_of_mat_idxs_vix,N,idxs_vix,device):
    '''
    loaded_model: keras model pre-trained
    mean_scaling_parameters: np.array 1-dim with the mean of training set for the model's parameters
    std_scaling_parameters: np.array 1-dim with the std of training set for the model's parameters
    parameters: torch.tensor 1-dim with the model's parameters (see ordering above), len=10 if beta_12 is included
    torch_mc: class of the "Volatility is (Mostly) Path-Dependent paper"
    mean_scaling_Rs:  np.array 1-dim with the mean of training set for the R's values
    std_scalings_Rs:  np.array 1-dim with the std of training set for the R's values
    list_of_mat_idxs_vix: list of the timestamps for the considered vix maturities
    N: int number of samples
    '''
    if idxs_vix is None:
        nbr_mat_vix=len(list_of_mat_idxs_vix)
        X=scale_and_get_X(parameters,torch_mc,mean_scaling_parameters,
                          std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,list_of_mat_idxs_vix,[k for k in range(nbr_mat_vix)],device)
    else:
        nbr_mat_vix=len(idxs_vix)
        X=scale_and_get_X(parameters,torch_mc,mean_scaling_parameters,
                          std_scaling_parameters,mean_scaling_Rs,std_scaling_Rs,list_of_mat_idxs_vix,idxs_vix,device)
  
        #X=X[:,idxs_vix,:]
        
 
    X=flatten_X(X).detach().cpu().numpy().T 
    approximated_vix=loaded_model(X).cpu().numpy()
    approximated_vix_scaled=scale_inverse_VIX(mean_scaling_VIX,std_scaling_VIX,approximated_vix,device)
    approximated_vix_scaled=approximated_vix_scaled.reshape(nbr_mat_vix,N,1)
    return 100*approximated_vix_scaled

def torch_futures(VIX):
    '''
    Axis=1 is the axis of the samples
    '''
    return torch.mean(VIX,axis=1)

def get_maximal_mat(idxs_vix,idxs_spx,tensor_spx_mat,tensor_vix_mat):
    'Since we want to simulate trajectories up to maturity we select as maximal maturity the maximal among the VIX and SPX'
    max_idx_vix=max(idxs_vix)
    max_idx_spx=max(idxs_spx)
    max_maturity=max(tensor_spx_mat[max_idx_spx],tensor_vix_mat[max_idx_vix])
    return max_maturity


def torch_payoff_call_vix_multiple_mat(VIX, normalized_strikes_vix,discount_vix,tensor_vix_mat,device):
    '''
    Here the VIX at all maturities should be fed into the function
    '''
    
    discount_vix=discount_vix.to(device)
    
    #torch_expy_rate_mat=torch.exp(-torch.mul(rate_vix,tensor_vix_mat))
    torch_expy_rate_mat=discount_vix
    return [torch.clamp(VIX[i]-normalized_strikes_vix[i].to(device),min=0)*torch_expy_rate_mat[i] for i in range(len(tensor_vix_mat))]

def torch_payoff_call_vix_select_mat(VIX, normalized_strikes_vix,discount_vix,tensor_vix_mat,idxs_vix,device):
    '''
    Here the VIX at the desired maturities should be fed into the function
    '''
    #torch_expy_rate_mat=torch.exp(-torch.mul(rate_vix,tensor_vix_mat))
    discount_vix=discount_vix.to(device)
    
    torch_expy_rate_mat=discount_vix
    torch_expy_rate_mat=torch_expy_rate_mat[idxs_vix]
    normalized_strikes_vix=[normalized_strikes_vix[k].to(device) for k in idxs_vix]
    return [torch.clamp(VIX[i]-normalized_strikes_vix[i],min=0)*torch_expy_rate_mat[i] for i in range(len(idxs_vix))]