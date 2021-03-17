import numpy as np
import h5py
import os
baseDir = os.getcwd()
# os.chdir("../Neuron/neuron_files/bbp/") # DO NOT keep this for when you want to run Allen
from neuron import h
# os.chdir(baseDir)
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd


def readParamsCSV(fileName):
    fields = ['Param name', 'Base value']
    df = pd.read_csv(fileName,skipinitialspace=True, usecols=fields)
    
    paramsList = [tuple(x) for x in df.values]
    return paramsList
    
run_file = 'run_model_cori.hoc'
stims_path = 'stims/stims_full.hdf5'
# CHANGE
orig_params = np.array(readParamsCSV("params/bbp_full_params_cc.csv"))[:,1]
orig_params = [float(orig_p) for orig_p in orig_params]
print(orig_params)
objectives_file = h5py.File('./objectives/multi_stim_bbp_full.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
ntimestep = 10000
dt = .02
def run_model(param_set, stim_name_list):
    h.load_file(run_file)
    volts_list = []
    for elem in stim_name_list:
        curr_stim = h5py.File(stims_path, 'r')[elem][:]
        total_params_num = len(param_set)
        timestamps = np.array([dt for i in range(ntimestep)])
        h.curr_stim = h.Vector().from_python(curr_stim)
        h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
        h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
        h.ntimestep = ntimestep
        h.runStim()
        out = h.vecOut.to_python()
        volts_list.append(out)
    return np.array(volts_list)

tV = target_volts_list = run_model(orig_params, opt_stim_list)
np.savetxt("targetVolts",tV, delimiter=",")