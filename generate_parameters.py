# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:29:40 2024

@author: Guido Gazzani
"""

import pickle
import pandas as pd
import torch
import os

cluster_flag=False
nbr_samples=10000
flag_beta_12=True

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def exp_kernel_GPU(t, lam):
    return lam * torch.exp(-lam * t)
def identity(x):
    return x
def squared(x):
    return x ** 2

def initialize_R(lam, dt=1/252,past_prices=None, max_delta=1000, transform=identity):
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




if flag_beta_12:
    names_params=['lam10','lam11','lam20','lam21','beta0','beta1','beta2','beta12','theta1','theta2']
else:
    names_params=['lam10','lam11','lam20','lam21','beta0','beta1','beta2','theta1','theta2']


def gen_rand_torch(r1,r2,a,b): #generate torch tensor of dimension (a,b) uniformly on [r1,r2]
    return (r2 - r1)*torch.rand(a, b)+r1

def combine_conditions(condition1,condition2):
    if len(condition1.shape)>1:
        condition1=condition1.squeeze(0)
    if len(condition2.shape)>1:
        condition2=condition2.squeeze(0)
    list_=[]
    for j in range(len(condition1)):
        if condition1[j]==condition2[j]:
            list_.append(condition1[j])
        else:
            list_.append(False)
    return torch.tensor(list_).unsqueeze(0)

if cluster_flag:
    print('Cannot import yf')
    from datetime import timedelta        
    dir_past_returns=r'/home/ag_cu-student/initial_values/codes_for_OM_data'
    os.chdir(dir_past_returns)
    with open('dictionary_initial_values_total.pkl', 'rb') as f:
        init_dict = pickle.load(f)
else:
    dir_past_returns=r'C:\Users\Guido Gazzani\Desktop\Paris\PDV\cluster_codes\codes_for_OM_data'
    os.chdir(dir_past_returns)
    with open('dictionary_initial_values_total.pkl', 'rb') as f:
        init_dict = pickle.load(f)
    



def get_initial_values_Rs_updated(lam1,lam2,spx):
    '''
    lam1, lam2 : tensors of length 2 positive values with lam_1,0>lam_1,1
    spx: output of the function get_data_yahoo_finance
    '''
    # Computes initial values from past prices
    R_init1 = initialize_R(lam1,dt=1/252, past_prices=spx, transform=identity)
    R_init2 = initialize_R(lam2,dt=1/252, past_prices=spx, transform=squared)
    return R_init1, R_init2


def get_samples_lambdas_condition_beta1(lower,upper,nbr_samples,idx_lambda,flag_beta12):
        list_lambda10=[]
        list_lambda11=[]        
        if idx_lambda==1:
            list_beta1=[]
            list_theta1=[]
        x=0
        for j in range(20): #trials
            lam10_train=gen_rand_torch(lower,upper,1,nbr_samples)
            lam11_train=gen_rand_torch(lower,upper,1,nbr_samples)
            condition1=lam10_train>lam11_train
            if idx_lambda==1:
                vol_of_vol_cap=10   ########################################################### specify here the cap of vol of vol
                beta1_train=gen_rand_torch(-0.2,-0.05,1,nbr_samples)
                theta1_train=gen_rand_torch(0,1,1,nbr_samples) 
                
                if flag_beta_12:
                    beta12_train=gen_rand_torch(0,0.25,1,nbr_samples)
                    model_vol_of_vol=torch.mul((1-theta1_train)*lam10_train+theta1_train*lam11_train,
                                               beta1_train.squeeze(0)+2*beta12_train.squeeze(0))
                else:
                    model_vol_of_vol=torch.mul((1-theta1_train)*lam10_train+theta1_train*lam11_train,beta1_train.squeeze(0))
                    
                condition2=torch.abs(model_vol_of_vol)<vol_of_vol_cap
                condition=combine_conditions(condition1,condition2)
                save_lam10=lam10_train[condition]
                save_lam11=lam11_train[condition]
                save_beta1=beta1_train[condition]
                save_theta1=theta1_train[condition]
            else:
                save_lam10=lam10_train[condition1]
                save_lam11=lam11_train[condition1]
            list_lambda10.append(save_lam10)
            list_lambda11.append(save_lam11)
            if idx_lambda==1:
                list_beta1.append(save_beta1)
                list_theta1.append(save_theta1)
            x=x+save_lam10.shape[0]
            if x>nbr_samples:
                break
        len_list=len(list_lambda10)
        lam10_train=torch.cat((list_lambda10[0],list_lambda10[1]))
        for j in range(len_list-2):
            lam10_train=torch.cat((lam10_train,list_lambda10[j+2]))
        lam11_train=torch.cat((list_lambda11[0],list_lambda11[1]))
        for j in range(len_list-2):
            lam11_train=torch.cat((lam11_train,list_lambda11[j+2]))

        lam10_train=lam10_train[:nbr_samples]
        lam11_train=lam11_train[:nbr_samples]
        if idx_lambda==1:
            beta1_train=torch.cat((list_beta1[0],list_beta1[1]))
            theta1_train=torch.cat((list_theta1[0],list_theta1[1]))
            for j in range(len_list-2):
                beta1_train=torch.cat((beta1_train,list_beta1[j+2]))
                theta1_train=torch.cat((theta1_train,list_theta1[j+2]))
            beta1_train=beta1_train[:nbr_samples]
            theta1_train=theta1_train[:nbr_samples]
            if flag_beta_12:
                return lam10_train, lam11_train,beta1_train, beta12_train.squeeze(0), theta1_train
            else:
                return lam10_train, lam11_train,beta1_train,theta1_train
        else:
            return lam10_train, lam11_train
