import numpy as np
import h5py
import os
os.chdir("neuron_files/allen/")
from neuron import h
os.chdir("../../")
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd

run_file = './neuron_files/allen/run_model_cori.hoc'
run_volts_path = './'
paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based.csv'
orig_params = nrnUtils.readBaseCSV(paramsCSV)
print(orig_params)

objectives_file = h5py.File('../results/485835016/allen485835016_objectives_passive.hdf5', 'r')
stims_path = '../results/485835016/stims_485835016_passive.hdf5'
target_volts_path = '../results/485835016/target_volts_485835016_passive.hdf5'
target_volts_hdf5 = h5py.File(target_volts_path, 'r')


opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
inds =  np.array([0,1,4,5,17])
opt_stim_name_list = [e.decode('ascii') for e in opt_stim_name_list]
opt_stim_name_list = np.array(opt_stim_name_list)[inds]

score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
target_volts_hdf5 = h5py.File(target_volts_path, 'r')

params_opt_ind = [0, 1, 6, 9, 13, 14, 15]
neg_index = 1

# Number of timesteps for the output volt.
ntimestep = 10000

def run_model(param_set, stim_name_list):
    h.load_file(run_file)
    volts_list = []
    for elem in stim_name_list:
        stims_hdf5 = h5py.File(stims_path, 'r')
        curr_stim = stims_hdf5[elem][:]
        total_params_num = len(param_set)
        dt = stims_hdf5[elem+'_dt']
        timestamps = np.array([dt for i in range(ntimestep)])
        h.curr_stim = h.Vector().from_python(curr_stim)
        h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
        h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
        h.ntimestep = ntimestep
        h.runStim()
        out = h.vecOut.to_python()        
        volts_list.append(out)
    return np.array(volts_list)

def evaluate_score_function(stim_name_list, target_volts_list, data_volts_list):
    def eval_function(target, data):
        return np.linalg.norm(target-data)**2 #  / np.linalg.norm(target)
    total_score = 0
    for i in range(len(stim_name_list)):
        curr_data_volt = data_volts_list[i]
        curr_target_volt = target_volts_list[i]
        curr_score = eval_function(curr_target_volt, curr_data_volt)
        total_score += curr_score
    return total_score

class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        params_ = nrnUtils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.opt_ind = params_opt_ind
        params_ = [params_[i] for i in self.opt_ind]
        self.orig_params = orig_params
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval)) for name, minval, maxval in params_]
        print("Params to optimize:", [(name, minval, maxval) for name, minval, maxval in params_])
        self.opt_stim_list = opt_stim_name_list
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        print("Init target volts")
        self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]
        
    def evaluate_with_lists(self, param_values):
        input_values = np.copy(self.orig_params)
        for i in range(len(param_values)):
            curr_opt_ind = self.opt_ind[i]
            input_values[curr_opt_ind] = param_values[i]
        input_values[neg_index] = -input_values[neg_index]
        data_volts_list = run_model(input_values, self.opt_stim_list)
        
        score = evaluate_score_function(self.opt_stim_list, self.target_volts_list, data_volts_list)
        return [score]
