import numpy as np
import h5py
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from pde import MemoryStorage

from methods.base import BaseConfig
from methods.io_things import add_data_to_h5dataset
from methods.plotting import plot_perm, plot_press, plot_cumulative_events_projection, plot_event_list, plot_events_projection
from methods.pore_press_calc_functions import Diffusion_with_Source_and_Gravity, Non_Uniform_Diffusion_with_Source_and_Gravity, test_wells, get_q_factors
from methods.seism_calc_functions import Micro_Seismic_Seeds, get_litostat_pressure, get_stress, \
    get_norm_and_shear_stress_on_seeds, check_colomb_criteria, get_raw_events, resample_raw_events, get_events, get_events_list, pad_events

params = BaseConfig()
params.load() # loading from existing params.yaml file. To dump some BaseConfig oject use its .dump(filename) mehod

# eq initial setup
eq = Diffusion_with_Source_and_Gravity(np.ones(params.shape), params)

perms_path = 'downscaled_models_04_16_2024__17_47_01.h5'


with h5py.File(perms_path, 'r') as f:
    mshape = f['perm'].shape
    nmodels = mshape[0]

idx_list = range(6339, nmodels)
# idx_list = range(6338, 6339)


pbar = tqdm(idx_list)

for ii in pbar:
    
    '''reading data'''
    pbar.set_postfix({'reading data: ': ii})

    with h5py.File(perms_path, 'r') as f:
        perm = f['perm'][ii]
        dens = f['dens'][ii]

    '''fluid dyn modeling'''
    pbar.set_postfix({'fluid dyn modeling': ii})

    eq.xi_field = eq.update_xi_field(perm) # updating permeability field

    p0 = eq.pore_ini_field
    q_factors = get_q_factors(perm, p0, params)
    eq.source_field = eq.update_source_field(0, q_factors) # updating source fields q_new = q * factor
    
    storage = MemoryStorage()
    res = eq.solve(p0, t_range=params.t_range, adaptive=True, tracker=[storage.tracker(1)])
    # res = eq.solve(p0, t_range=params.t_range, adaptive=True, tracker=['progress', 'plot', storage.tracker(1)]) # use it for testing
    pore_press = np.stack(storage.data, axis=0) # 4d np array

    '''seismic modeling'''
    pbar.set_postfix({'seismic modeling': ii})
    seeds = Micro_Seismic_Seeds(params) # creating microseismic seeds obj
    tan_phi, C, norms = seeds.tan_phi_rvs, seeds.C_rvs, seeds.norms_rvs

    lithostat_pressure = get_litostat_pressure(params, dens)
  
    stress = get_stress(params, lithostat_pressure) # litostatic stress tensor based on density model
    sigma_n, tau = get_norm_and_shear_stress_on_seeds(stress, norms) # stresses without pore pressure

    colomb_pass = check_colomb_criteria(params, pore_press, tan_phi, C, sigma_n, tau) # (dim broadcasting) 

    raw_events = get_raw_events(params, colomb_pass) # num of seeds passed in every cell
    events_dens = resample_raw_events(params, raw_events) # event probability (raw_events normalized to target num)
    events = get_events(params, events_dens) # events

    events_list = get_events_list(events) # list of events (time, x, y, d, Mag)

    events_by_time = np.sum(events, axis=(1,2,3)) # events number at every step
    tot_events = np.sum(events)

    '''saving'''
    pbar.set_postfix({'saving data': ii})
    ev_len_with_pad = int(1.1 * params.target_events_num) # 10% padding for data shape consistency
    padded_events = pad_events(events_list, ev_len_with_pad) # fills extra lines with -1

    add_data_to_h5dataset(perms_path, ii, padded_events, 'events')
    add_data_to_h5dataset(perms_path, ii, pore_press[-1], 'pore')
    add_data_to_h5dataset(perms_path, ii, np.cumsum(events_dens, axis=0)[-1], 'ev_dens') # cumulative seismic density al the last step  

# at last writing params to file just for the case
fname_for_params = f'params_{perms_path.split('.')[0]}.yaml'
params.dump(fname_for_params)


# PermissionError: [Errno 13] Unable to synchronously open file (unable to open file: name = 'downscaled_models_04_16_2024__17_47_01.h5', errno = 13, error message = 'Permission denied', flags = 1, o_flags = 2)
