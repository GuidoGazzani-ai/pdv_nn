# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:20:02 2023

@author: Guido Gazzani
"""

import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import fsolve
from tqdm.auto import tqdm

torch_normal_law = torch.distributions.normal.Normal(0, 1.)
pi = torch.tensor(np.pi)
dt = 1/252
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def exp_kernel_GPU(t, lam, c=1):
    return c * lam * torch.exp(-lam * t)
def identity(x):
    return x
def squared(x):
    return x ** 2

def initialize_R(lam, past_prices=None, max_delta=1000, transform=identity):
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

class TorchMonteCarloExponentialModel:
    def __init__(self, lam1, lam2, R_init1, R_init2, betas, theta1=0., theta2=0., parabolic=0., parabolic_offset=0.,
                 S0=1,
                 maturity=30 * dt,
                 timestep_per_day=1, N=int(1e5), vix_N=int(1e3), fixed_seed=False, seed_root=3748192728942,
                 vol_cap=1.5, device=device):
        """
        Used to simulate the prices of an asset whose volatility follows the dynamic:
        $$
        \frac{dS_t}{S_t} = \sigma_t dW_t \\
        \sigma_t = \beta_0 + \beta_1 R_{1,t} + \beta_2 \sqrt{R_{2,t}} + \beta_{1,2} R_{1}^2{\bf 1}_{R_1\geq c} \\
        R_{i,t} = (1-\theta_i) R_{i,0,t} + \theta_i R_{i,1,t}, i\in\{1,2\} \\
        dR_{1,j,t} = \lambda_{1,j} (\sigma_t dW_t - R_{1,j,t}), j\in \{0,1\} \\
        dR_{2,j,t} = \lambda_{2,j}  (\sigma_t^2 dt - R_{2,j,t}), j\in \{0,1\}
        $$
        :param lam1: array-like or tensor of size 2
        :param lam2: array-like or tensor of size 2
        :param R_init1: array-like or tensor of size 2
        :param R_init2: array-like or tensor of size 2
        :param betas: array-like or tensor of size 3
        :param S0: float. Default 1
        :param theta1: float or tensor of size 0
        :param theta2: float or tensor of size 0
        :param maturity: float. Maturity (in years) of the simulation
        :param timestep_per_day: int. Number of steps per day for the montecarlo simulation
        :param N: int. number of paths for the MC simulation
        :param vix_N: int. number of sub-paths for the computation of VIX
        :param fixed_seed: bool. If True, uses the seed_root as the initial seed of dW, then inc
        :param seed_root: int, first seed for the MC simulation
        :param vol_cap: float. the instanteneous volatility is capped at this value
        :param device: torch.device or str. Device on which the computation is done
        :param parabolic: float or tensor of size 0. Default 0. value of the parabolic coefficient $\beta_{1,2}$
        :param parabolic_offset: float or tensor of size 0. Default 0, Value of the offset $c$ before the parabolic term
        """
        self.device = torch.device(device)
        self.timestep_per_day = timestep_per_day  #integer number of time steps you have per day
        self.timestep = torch.tensor(dt / timestep_per_day, device=self.device) #size of actual yearly time step
        self.maturity = maturity
        self.T = self.index_of_timestamp(self.maturity)  # number of integers
        self.N = N
#         self.R_init1 = convert_to_tensor(R_init1).to(self.device)
#         self.R_init2 = convert_to_tensor(R_init2).to(self.device)
#         self.lam1 = convert_to_tensor(lam1).to(self.device)
#         self.lam2 = convert_to_tensor(lam2).to(self.device)
#         self.theta1 = convert_to_tensor(theta1).to(self.device)
#         self.theta2 = convert_to_tensor(theta2).to(self.device)
#         self.betas = convert_to_tensor(betas).to(self.device)
        self.R_init1 = R_init1.to(self.device)
        self.R_init2 = R_init2.to(self.device)
        self.lam1 = lam1.to(self.device)
        self.lam2 = lam2.to(self.device)
        self.theta1 = theta1.to(self.device)
        self.theta2 = theta2.to(self.device)
        self.betas = betas.to(self.device)
        self.vix_N = vix_N
        self.vix_steps = self.index_of_timestamp(30 / 365)
        self.S0 = S0
        self.fixed_seed = fixed_seed
        self.seed_root = seed_root
        self.vol_cap = vol_cap
        self.parabolic = convert_to_tensor(parabolic, array=False).to(self.device)
        self.parabolic_offset = convert_to_tensor(parabolic_offset, array=False).to(self.device)

    def index_of_timestamp(self, t):
        """
        :param t: float or torch.tensor
        :return: int, index of the timestamp t
        """
        return torch.round(t / self.timestep).to(int)

    def compute_vol(self, R_1, R_2):
        """
        computes volatility
        :param R_1: torch.tensor of the same size as R_2
        :param R_2: torch.tensor of the same size as R_1
        :return: volatity of the same size as R_1 or R_2
        """
        vol = self.betas[0] + self.betas[1] * R_1 + self.betas[2] * torch.sqrt(R_2) + self.parabolic * (
                R_1 - self.parabolic_offset).clamp(
            min=0)**2
        if self.vol_cap is None:
            return vol
        return vol.clamp(max=self.vol_cap)

    def compute_R(self, i=1):
        """
        computes $R_i = (1-\theta_i) R_{i,0} + \theta_i R_{i,1}, $
        :param i: int 1 or 2
        :return: return tensor of all $R_i$ from the simulation
        """
        if i == 1:
            return self.R1_array[:, 0] * (1 - self.theta1.cpu()) + self.R1_array[:, 1] * self.theta1.cpu()
        
        elif i == 2:
            return self.R2_array[:, 0] * (1 - self.theta2.cpu()) + self.R2_array[:, 1] * self.theta2.cpu()
        else:
            raise ValueError('i in (1,2) only')

    def _simulate(self, n_timesteps, n_paths, S0, R1_0, R2_0, seed_root=0, save_R=False):
        """
        Simulates n_paths over n_timesteps of the dynamics
        :param n_timesteps: int, number of timestepss
        :param n_paths: int, number of paths
        :param S0: float or tensor of size n_paths, initial value(s) of S
        :param R1_0: float or tensor of size (n_paths, 2), initial value(s) of R_{1,j}
        :param R2_0: float or tensor of size (n_paths, 2) initial value(s) of R_{2,j}
        :param seed_root: in
        :param save_vol_only: bool. If True, only keeps the tensor of volatility(to save memory). Otherwise, saves also S, R1 and R2.
        :return: tensor of volatility of shape (n_timesteps+1, n_paths) if save_vol_only. Otherwise, returns also S, R1 and R2 tensors.
        """
        
        
        
        r1 = R1_0.to(self.device)
        r2 = R2_0.to(self.device)
        vol_array = torch.zeros((n_timesteps + 1, n_paths), device=self.device)
        if save_R:
            R1_array = torch.zeros((n_timesteps + 1, 2, n_paths), device=self.device)  # ()
            R2_array = torch.zeros((n_timesteps + 1, 2, n_paths), device=self.device)  # (self.device)
        S_array = torch.zeros((n_timesteps + 1, n_paths), device=self.device)
        S_array[0] = S0
        #brownian_increments = self.timestep.sqrt() * torch.randn(n_timesteps, n_paths, device=self.device) #new
        for t in range(n_timesteps):
            R1 = (1 - self.theta1) * r1[0] + self.theta1 * r1[1]
            R2 = (1 - self.theta2) * r2[0] + self.theta2 * r2[1]
            
            vol = self.compute_vol(R1, R2)
            vol_array[t] = vol
            if save_R:
                R1_array[t].copy_(r1.to(self.device))
                R2_array[t].copy_(r2.to(self.device)) #=r1.cpu()
            if self.fixed_seed:
                torch.manual_seed(seed_root + t)
            brownian_increment = self.timestep.sqrt()*torch.randn(n_paths, device=self.device)
            increment = vol * brownian_increment#s #[t]
            for j in range(2):
                r1[j] = (torch.exp(-self.lam1[j] * self.timestep) * (r1[j] + self.lam1[j] * increment)).clone()
                r2[j] = (torch.exp(-self.lam2[j] * self.timestep) * (r2[j] + self.lam2[j] * vol ** 2 * self.timestep)).clone()
            S_array[t+1] = S_array[t].clone() * torch.exp(increment - 0.5*vol**2*self.timestep)
        R1 = (1 - self.theta1) * r1[0] + self.theta1 * r1[1]
        R2 = (1 - self.theta2) * r2[0] + self.theta2 * r2[1]

        
        vol_array[n_timesteps] = self.compute_vol(R1, R2)
        if save_R:
            R1_array[n_timesteps].copy_(r1.to(self.device))
            R2_array[n_timesteps].copy_(r2.to(self.device))
            return S_array, vol_array, R1_array, R2_array
        else:
            return S_array, vol_array

        
    def simulate(self, save_R=True): 
        """
        simulates until maturity.
        :return:
        """
        if len(self.R_init1.shape) == 1:
            R1_0 = self.R_init1.unsqueeze(1).repeat_interleave(self.N, dim=1)
            R2_0 = self.R_init2.unsqueeze(1).repeat_interleave(self.N, dim=1)
        else:
            R1_0 = self.R_init1
            R2_0 = self.R_init2
       
        if save_R:
            self.S_array, self.vol_array, self.R1_array, self.R2_array = \
                self._simulate(self.T, self.N, R1_0=R1_0, R2_0=R2_0, S0=self.S0, seed_root=self.seed_root,
                               save_R=save_R)
        else:
            self.S_array, self.vol_array = \
                self._simulate(self.T, self.N, R1_0=R1_0, R2_0=R2_0, S0=self.S0, seed_root=self.seed_root,
                               save_R=save_R)
                
    def compute_option_price(self, strikes=None, option_maturity=None, scaled_fwd=None, discount=None,
                                 return_future=False,var_reduction=True,flag_put=False, sigma0=0.1):
            """
            Computes the call option prices onn the underlying
            :param strikes: float or torch.tensor of size n_K
            :param option_maturity: maturity of the option
            :param return_future: if True, returns the future/forward
            :param var_reduction: if True (only for S), uses theta-gamma method to reduce variance.
            :param sigma0: float. Default 0.1. Value of $\sigma_0$ for the variance reduction
            :return: tuple of strikes and option prices. both of size n_K
            """
            payoff = torch_payoff_call
            maturity = option_maturity
            index = int(torch.ceil(maturity / self.timestep))
            array = self.S_array
            S = array[index]
            strikes = convert_to_tensor(strikes, array=True).to(self.device)
            payoff_values = payoff(S,strikes, maturity,scaled_fwd,discount)
            expected_value = payoff_values.mean(axis=0)
            
            if var_reduction:
                expected_value_classic = expected_value
                black_scholes_price_0 = BS_priceGPU_general(strikes, maturity,array[0,0].clone(),sigma0,scaled_fwd,discount)#.float() 
                time_to_maturity = torch.arange(maturity, 0, -self.timestep, device=self.device)
                f_tT = self.S_array[:index]
                f_per_strike = f_tT.unsqueeze(0)
                sigma_t = self.vol_array[:index]
                gammas = BS_gammaGPU_general2(strikes[:, np.newaxis, np.newaxis], time_to_maturity[:, np.newaxis], f_per_strike, sigma0,scaled_fwd) 
                PnL = (scaled_fwd*discount)*(sigma_t ** 2 - sigma0 ** 2) * f_per_strike ** 2 * gammas
                pnl_per_simulation =1 / 2 *PnL.sum(axis=1)*self.timestep
                pnl_expectancy = pnl_per_simulation.mean(axis=1)
                expected_value = pnl_expectancy + black_scholes_price_0
                expected_value[expected_value <= 0] = expected_value_classic[expected_value <= 0]
            return expected_value        
                
    def compute_option_price_mod(self, strikes=None, option_maturity=None, scaled_fwd=None, discount=None,
                             return_future=False,var_reduction=True,flag_put=False, sigma0=0.1):
        """
        Computes the call option prices onn the underlying
        :param strikes: float or torch.tensor of size n_K
        :param option_maturity: maturity of the option
        :param return_future: if True, returns the future/forward
        :param var_reduction: if True (only for S), uses theta-gamma method to reduce variance.
        :param sigma0: float. Default 0.1. Value of $\sigma_0$ for the variance reduction
        :return: tuple of strikes and option prices. both of size n_K
        """
        if flag_put:
            payoff = torch_payoff_put
        else:
            payoff = torch_payoff_call
        maturity = option_maturity
        index = int(torch.ceil(maturity / self.timestep))
        array = self.S_array
        S = array[index]
        strikes = convert_to_tensor(strikes, array=True).to(self.device)
        payoff_values = payoff(S,strikes, maturity,scaled_fwd,discount)
        std_no_vr=payoff_values.std(axis=0)
        expected_value = payoff_values.mean(axis=0)
        
        if var_reduction:
            expected_value_classic = expected_value
            black_scholes_price_0 = BS_priceGPU_general(strikes, maturity,array[0,0].clone(),sigma0,scaled_fwd,discount)#.float() 
            time_to_maturity = torch.arange(maturity, 0, -self.timestep, device=self.device)
            f_tT = self.S_array[:index]
            f_per_strike = f_tT.unsqueeze(0)
            sigma_t = self.vol_array[:index]
            gammas = BS_gammaGPU_general2(strikes[:, np.newaxis, np.newaxis], time_to_maturity[:, np.newaxis], f_per_strike, sigma0,scaled_fwd) 
            PnL = (scaled_fwd*discount)*(sigma_t ** 2 - sigma0 ** 2) * f_per_strike ** 2 * gammas
            pnl_per_simulation =1 / 2 *PnL.sum(axis=1)*self.timestep
            pnl_expectancy = pnl_per_simulation.mean(axis=1)
            std_vr_call=pnl_per_simulation.std(axis=1)
            expected_value = pnl_expectancy + black_scholes_price_0
            expected_value[expected_value <= 0] = expected_value_classic[expected_value <= 0]
            
        if flag_put:
            expected_value_put=expected_value+strikes*discount-array[0,0].clone()*(scaled_fwd*discount)
            return expected_value_put,payoff_values.mean(axis=0),std_no_vr,std_vr_call
        
        else:
            return expected_value,payoff_values.mean(axis=0),std_no_vr,std_vr_call
        
    def compute_option_price_mod2(self, strikes=None, option_maturity=None, scaled_fwd=None, discount=None,
                             return_future=False,var_reduction=True, sigma0=0.1):
        """
        Computes the call option prices onn the underlying
        :param strikes: float or torch.tensor of size n_K
        :param option_maturity: maturity of the option
        :param return_future: if True, returns the future/forward
        :param var_reduction: if True (only for S), uses theta-gamma method to reduce variance.
        :param sigma0: float. Default 0.1. Value of $\sigma_0$ for the variance reduction
        :return: tuple of strikes and option prices. both of size n_K
        """
        mask=strikes<=1.2
        
        maturity = option_maturity
        index = int(torch.ceil(maturity / self.timestep))
        array = self.S_array
        S = array[index]
        strikes = convert_to_tensor(strikes, array=True).to(self.device)
        
        payoff_values_put = torch_payoff_put(S,strikes[mask], maturity,scaled_fwd,discount)
        payoff_values_call = torch_payoff_call(S,strikes[~mask], maturity,scaled_fwd,discount)
        
        std_no_vr_put,std_no_vr_call=payoff_values_put.std(axis=0),payoff_values_call.std(axis=0)
        
        
        expected_value_put = payoff_values_put.mean(axis=0)
        expected_value_call = payoff_values_call.mean(axis=0)
        
        
        if var_reduction:
            expected_value_classic_call = expected_value_call
            expected_value_classic_put = expected_value_put
            
            ### call
            black_scholes_price_0 = BS_priceGPU_general(strikes[~mask], maturity,array[0,0].clone(),sigma0,scaled_fwd,discount)#.float() 
            time_to_maturity = torch.arange(maturity, 0, -self.timestep, device=self.device)
            f_tT = self.S_array[:index]
            f_per_strike = f_tT.unsqueeze(0)
            sigma_t = self.vol_array[:index]
            gammas = BS_gammaGPU_general2(strikes[~mask, np.newaxis, np.newaxis], time_to_maturity[:, np.newaxis], f_per_strike, sigma0,scaled_fwd) 
            PnL = (scaled_fwd*discount)*(sigma_t ** 2 - sigma0 ** 2) * f_per_strike ** 2 * gammas
            pnl_per_simulation =1 / 2 *PnL.sum(axis=1)*self.timestep
            pnl_expectancy = pnl_per_simulation.mean(axis=1)
            std_vr_call=pnl_per_simulation.std(axis=1)
            expected_value_call = pnl_expectancy + black_scholes_price_0
            expected_value_call[expected_value_call <= 0] = expected_value_classic_call[expected_value_call <= 0]
            
            ### put
            black_scholes_price_0 = BS_priceGPU_general(strikes[mask], maturity,array[0,0].clone(),sigma0,scaled_fwd,discount)#.float() 
            gammas = BS_gammaGPU_general2(strikes[mask, np.newaxis, np.newaxis], time_to_maturity[:, np.newaxis], f_per_strike, sigma0,scaled_fwd) 
            PnL = (scaled_fwd*discount)*(sigma_t ** 2 - sigma0 ** 2) * f_per_strike ** 2 * gammas
            pnl_per_simulation =1 / 2 *PnL.sum(axis=1)*self.timestep
            pnl_expectancy = pnl_per_simulation.mean(axis=1)
            std_vr_put=pnl_per_simulation.std(axis=1)
            expected_value_put = pnl_expectancy + black_scholes_price_0+strikes[mask]*discount-array[0,0].clone()*(scaled_fwd*discount)
            expected_value_put[expected_value_put <= 0] = expected_value_classic_put[expected_value_put <= 0]
            
        expected_value=torch.cat((expected_value_put,expected_value_call))

        return expected_value,torch.cat((payoff_values_put.mean(axis=0), payoff_values_call.mean(axis=0))),torch.cat((std_vr_put,std_vr_call)),torch.cat((std_no_vr_put,std_no_vr_call))
    
    def _compute_vix_nested(self, vix_index, idxs=None, n_subpaths=None):
        """
        computes VIX via nested MC for the paths given by idxs
        :param vix_index: int. index corresponding to the vix maturity
        :param idxs: int or torch.tensor. paths whose vix is computed
        :param n_subpaths: int. number of subpaths used.
        :return:
        """
        if idxs is None:
            idxs = torch.arange(self.N)
        if n_subpaths is None:
            n_subpaths = self.vix_N
        size = len(idxs)
        #vix_index = int(torch.ceil(vix_maturity / self.timestep))

        S0 = torch.repeat_interleave(self.S_array[vix_index, idxs], n_subpaths)
        R1_0 = torch.repeat_interleave(self.R1_array[vix_index, :, idxs], n_subpaths, dim=1)
        R2_0 = torch.repeat_interleave(self.R2_array[vix_index, :, idxs], n_subpaths, dim=1)

        # print(S0.shape)
        # print(R1_0.shape)
        # print(R2_0.shape)
        # print(n_subpaths)
        # print(size)
        
        # print('VIX STEPS:',self.vix_steps)
        _, nested_vol_array = \
            self._simulate(self.vix_steps, size * n_subpaths, S0=S0, R1_0=R1_0, R2_0=R2_0, save_R=False)
        nested_vol_array = nested_vol_array.reshape((self.vix_steps + 1, size, -1))
        return (nested_vol_array ** 2).mean(dim=(0, 2))

    def compute_vix(self, vix_maturity, subset=None, n_batch=1):
        """
        compute the VIX via nested MC for each path at timestep vix_maturity
        :param vix_maturity: float,
        :param n_batch: int. Divides the paths in batches to compute VIX. This allows to save memory.
        :return: tensor of size self.N. VIX per path
        """
        if subset is None:
            subset = torch.arange(self.N)
        elif isinstance(subset, int):
            subset = torch.arange(subset)
        subset = convert_to_tensor(subset).sort().values
        n = subset.shape[0]
        vix_index = int(torch.ceil(vix_maturity / self.timestep))
        idxs = torch.linspace(0, n, n_batch + 1, dtype=torch.int)
        vix_squared = torch.zeros(self.N, device=self.device)
        for i in tqdm(range(n_batch),desc='By batch'):
            vix_squared[idxs[i]:idxs[i + 1]] = self._compute_vix_nested(vix_index,
                                                                        idxs=subset[idxs[i]: idxs[i + 1]])
        return vix_squared.sqrt()
    
    def compute_vix_path_jordan(self, path_id, n_subpaths=None, n_batch=1):
        if n_subpaths is None:
            n_subpaths = self.vix_N
        S0 = self.S_array[1::self.timestep_per_day, path_id]
        R1_0 = self.R1_array[1::self.timestep_per_day, :, path_id].T
        R2_0 = self.R2_array[1::self.timestep_per_day, :, path_id].T
        idxs = torch.linspace(0, S0.shape[0], steps=n_batch + 1, dtype=torch.int64)
        vix = torch.zeros(S0.shape[0])
        for i in range(len(idxs) - 1):
            size = idxs[i + 1] - idxs[i]
            S = S0[idxs[i]:idxs[i + 1]]
            R1 = R1_0[:, idxs[i]:idxs[i + 1]]
            R2 = R2_0[:, idxs[i]:idxs[i + 1]]

            S = torch.repeat_interleave(S, n_subpaths)
            R1 = torch.repeat_interleave(R1, n_subpaths, dim=1)
            R2 = torch.repeat_interleave(R2, n_subpaths, dim=1)
            _, nested_vol_array = \
                self._simulate(self.vix_steps, size * n_subpaths, S0=S, R1_0=R1, R2_0=R2,
                               save_R=False)
            nested_vol_array = nested_vol_array.reshape((self.vix_steps + 1, size, -1))

            vix[idxs[i]:idxs[i + 1]] = (nested_vol_array ** 2).mean(dim=(0, 2)).sqrt().cpu()

        return vix
    
    def compute_vix_path(self, path_id, n_subpaths=None, n_batch=1):
        if n_subpaths is None:
            n_subpaths = self.vix_N
        S0 = self.S_array[:, path_id]
        R1_0 = self.R1_array[:, :, path_id].T
        R2_0 = self.R2_array[:, :, path_id].T
        idxs = torch.linspace(0, S0.shape[0], steps=n_batch + 1, dtype=torch.int64)
        vix = torch.zeros(S0.shape[0])
        for i in range(len(idxs) - 1):
            size = idxs[i + 1] - idxs[i]
            S = S0[idxs[i]:idxs[i + 1]]
            R1 = R1_0[:, idxs[i]:idxs[i + 1]]
            R2 = R2_0[:, idxs[i]:idxs[i + 1]]

            S = torch.repeat_interleave(S, n_subpaths)
            R1 = torch.repeat_interleave(R1, n_subpaths, dim=1)
            R2 = torch.repeat_interleave(R2, n_subpaths, dim=1)
            _, nested_vol_array = \
                self._simulate(self.vix_steps, size * n_subpaths, S0=S, R1_0=R1, R2_0=R2,
                               save_R=False)
            nested_vol_array = nested_vol_array.reshape((self.vix_steps + 1, size, -1))

            vix[idxs[i]:idxs[i + 1]] = (nested_vol_array ** 2).mean(dim=(0, 2)).sqrt().cpu()

        return vix

    
    

def BS_priceGPU_general(K, T, S, vol,scaled_fwd,discount):
    """

    :param K: torch.Tensor of shape k
    :param T: float, maturity
    :param F: torch.Tensor of shape N or float, forward price
    :param vol: float,
    :return: torch.Tensor of shape k x N containing the Black Gamma
    """
    T = convert_to_tensor(T, array=False)
    vol = convert_to_tensor(vol, array=False)
    K = convert_to_tensor(K, array=True)
    S = convert_to_tensor(S, array=True)
    s = vol * T.sqrt()
    d1 = (torch.log(torch.true_divide(S, K))+torch.log(scaled_fwd))/s +0.5*s 
    
    return (discount*scaled_fwd)*torch_normal_law.cdf(d1)*S-torch_normal_law.cdf(d1-s)*K*discount


def BS_gammaGPU_general2(K, T, S, vol,scaled_fwd):
    """

    :param K: torch.Tensor of shape k
    :param T: float, maturity
    :param F: torch.Tensor of shape N or float, forward price
    :param vol: float,
    :return: torch.Tensor of shape k x N containing the Black Gamma
    """
    T = convert_to_tensor(T, array=False)
    vol = convert_to_tensor(vol, array=False)
    K = convert_to_tensor(K, array=True)
    S = convert_to_tensor(S, array=True)
    s = vol * T.sqrt()
    d1 = (torch.log(torch.true_divide(S, K))+torch.log(scaled_fwd))/s +0.5*s
    norm_pdf = torch.exp(-d1 ** 2 / 2) / torch.sqrt(2 * pi)
    return norm_pdf/(S*s)



def torch_payoff_call(S, K, maturity, scaled_fwd,discount):
    """
    Recall that for rate we mean \int_0^maturity r_s ds, and for divi the constant dividend q.
    """
    K = convert_to_tensor(K, array=True)
    S=S*scaled_fwd  #divi*maturity
    return torch.clamp(S[:, np.newaxis] - K, min=0)*discount


def torch_payoff_put(S, K, maturity, scaled_fwd,discount):
    """
    Recall that for rate we mean \int_0^maturity r_s ds, and for divi the constant dividend q.
    """
    K = convert_to_tensor(K, array=True)
    S=S*scaled_fwd  #divi*maturity
    return torch.clamp(K-S[:, np.newaxis], min=0)*discount

def torch_payoff_call_jordan(S, K):
    K = convert_to_tensor(K, array=True)
    return torch.clamp(S[:, np.newaxis] - K, min=0)



def convert_to_tensor(x, array=True):
    """
    :param x: float or array-like
    :param float: bool. If float is False, then x always return a tensor of dimension at least 1.
    :return: tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
    if array:
        if x.dim() == 0:
            x.unsqueeze_(0)
    return x


def convert_to_tensor_arr(x, array=True):
    """
    :param x: float or array-like
    :param float: bool. If float is False, then x always return a tensor of dimension at least 1.
    :return: tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
        #torch.tensor(x).float()
    if array:
        if x.dim() == 0:
            x.unsqueeze_(0)
    return x

def numpy_arrays_to_tensors(numpy_list,device):
    """
    Convert a list of NumPy arrays to a list of PyTorch tensors.

    Parameters:
    - numpy_list (list): A list containing NumPy arrays.

    Returns:
    - torch_tensor: torch.tensor containing PyTorch tensors.
    """
    torch_list = [torch.from_numpy(arr).to(device) for arr in numpy_list]
    
    def filter_tensors_by_length(tensor_list, target_length):
        filtered_tensors = [tensor for tensor in tensor_list if len(tensor) == target_length]
        return filtered_tensors
    
    maximal_len=max(list(map(lambda y: len(y),torch_list)))
    torch_list = torch.stack(filter_tensors_by_length(torch_list,maximal_len)).to(device)
    return torch_list


def phi(x): ## Gaussian density
    return np.exp(-x*x/2.)/np.sqrt(2*np.pi)   

    

def find_ivol_bnp(price, spot, strike, T, scaled_fwd,discount):
    'This Implied Vol function is different with respect to the previous:'
    'As r>0 one feeds \int_0^T r_s ds; i.e. continous deterministic interest rate'

    def BS_price(sigma):
        d_1= 1/(sigma*np.sqrt(T))*(np.log(spot*scaled_fwd/strike)+(sigma**2/2)*T)
        d_2= d_1-sigma*np.sqrt(T)
        
        N_1= norm.cdf(d_1) #scipy.stats.norm.cdf
        N_2= norm.cdf(d_2) #scipy.stats.norm.cdf
        
        return N_1*spot*(scaled_fwd*discount)-N_2*strike*discount - price
    
    root = fsolve(BS_price, 1)[-1] 
    return root




def iv_black(price,strike, maturity, future,discount):
    def black_call_aux(vola):
        'Function to price with Black Call options on Futures'
        # Calculate components of the Black-Scholes formula
        time_to_maturity = maturity
        discount_factor = discount
        
        d1 = (np.log(future / strike) -np.log(discount_factor) + ((vola ** 2) / 2) * time_to_maturity) / (vola * np.sqrt(time_to_maturity))
        
        # Use cumulative distribution function from scipy.stats to get N(d1) and N(d2)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d1 - vola * np.sqrt(time_to_maturity))
        
        # Calculate call option price using Black-Scholes formula
        call_price = discount_factor*(future*nd1 - strike*nd2)
        return call_price-price
    root = fsolve(black_call_aux, 1)[-1]
    return root
