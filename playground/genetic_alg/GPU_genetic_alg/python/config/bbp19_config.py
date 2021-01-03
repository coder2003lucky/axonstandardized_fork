import csv
import pandas as pd
import os
import numpy as np
import h5py

paramsCSV = '../params/params_bbp_full.csv'
#orig_params = np.array(np.array(nrnUtils.readParamsCSV(paramsCSV))[:,1], dtype=np.float64)
orig_params = h5py.File('../params/params_bbp_full.hdf5', 'r')['orig_full'][0]
scores_path = '../scores/'
objectives_file = h5py.File('../objectives/multi_stim_bbp_full.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
stims_path = '../stims/stims_full.hdf5'
stim_file = h5py.File(stims_path, 'r')
#target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
#target_volts_hdf5 = h5py.File(target_volts_path, 'r')
#params_opt_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
params_opt_ind = np.arange(24) 
model_dir = '..'
data_dir = model_dir+'/Data/'
run_dir = '../bin'
vs_fn = '/tmp/Data/VHotP'

# Number of timesteps for the output volt.
ntimestep = 10000

