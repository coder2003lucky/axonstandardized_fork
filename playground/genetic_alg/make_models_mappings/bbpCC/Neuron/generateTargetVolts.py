import numpy as np
import h5py
import os
os.chdir("neuron_files/bbp/") # DO NOT keep this for when you want to run Allen
from neuron import h
os.chdir("../../")
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd

run_file = 'runModel.hoc'
stims_path = '../stims/' + inputs['stim_file'] + '.hdf5'
# CHANGE
orig_params = h5py.File('./params/params_' + model + '_' + peeling + '.hdf5', 'r')['orig_passive'][0]
objectives_file = h5py.File('../objectives/multi_stim_without_sensitivity_'+ model + '_' + peeling + '_' + date + '_stims.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]

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

if __name__ == main:
    target_volts_list = run_model(orig_params, opt_stim_list)