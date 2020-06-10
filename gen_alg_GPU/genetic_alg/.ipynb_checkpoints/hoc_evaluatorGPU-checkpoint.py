import numpy as np
import h5py
from neuron import h
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd
import os
import subprocess
import time

import os

import numpy as np
import os
import subprocess
import shutil
import bluepyopt as bpop
import struct
import time
import pandas as pd



model='bbp'
peeling='potassium'
date='04_08_2020'
params_opt_ind=[5,7,12,13,17]
run_file = './run_model_cori.hoc'
run_volts_path = '../../../run_volts/run_volts_' + model + "_" + peeling
paramsCSV = '../../../param_stim_generator/params_reference/params_'+ model + '_' + peeling +'.csv'
orig_params = h5py.File('../../../params/params_' + model + '_' + peeling + '.hdf5', 'r')['orig_' + peeling][0]
scores_path = '../scores/'



model_dir = 'pyNeuroGPU_unix'
param_file ='./params/gen.csv'               #What is gen.csv? does it matter?
data_dir = model_dir+'/Data/'
params_table = data_dir + 'opt_table.csv'    #bbp template ORIG
run_dir = '../bin'
orig_volts_fn = data_dir + 'exp_data.csv' #ORIG volts
vs_fn = model_dir + '/Data/VHotP'
times_file_path = model_dir + '/Data/times.csv'
nstims = 2
target_volts = np.genfromtxt(orig_volts_fn)
times =  np.cumsum(np.genfromtxt(times_file_path,delimiter=','))
#nCpus =  multiprocessing.cpu_count()
# TO DO: set link to correct objectives_file
# objectives_file = h5py.File('./objectives/multi_stim_without_sensitivity_' + model +  '_' + peeling \
# + '_1_0_20_stims_no_v_init.hdf5', 'r')
#objectives_file = h5py.File('./objectives/multi_stim_without_sensitivity_bbp_sodium_1_0_20_stims_no_v_init.hdf5', 'r')
objectives_file = h5py.File('./objectives/multi_stim_without_sensitivity_' + model \
+ '_' + peeling +'_'+ date + '_stims.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
stims_path = '../../../stims/stims_full.hdf5'
#params_opt_ind = [9, 10, 14, 17, 18, 22]


custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']

# Number of timesteps for the output volt.
ntimestep = 10000

# Value of dt in miliseconds
dt = 0.02

def run_model(stim_ind, params):
    print("running stim ind" + str(stim_ind))
    volts_fn = vs_fn + str(stim_ind) + '.h5'
    #volts_fn = vs_fn + str(stim_ind) + '.dat'
    if os.path.exists(volts_fn):
        os.remove(volts_fn)
        #pass
    #path = "./sleep.sh"
    #p_object = subprocess.Popen(path, shell=True)
    p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)])
    print(os.path.exists('../bin/neuroGPU'))
    #p_object = subprocess.Popen(['../bin/neuroGPU2',str(stim_ind)])

    return p_object


def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)


def evaluate_score_function(stim_name_list, target_volts_list, data_volts_list, weights):
    def eval_function(target, data, function, dt):
        if function in custom_score_functions:
            score = getattr(sf, function)(target, data, dt)
        else:
            score = sf.eval_efel(function, target, data, dt)
        return score
    def normalize_single_score(newValue, transformation):
        # transformation contains: [bottomFraction, numStds, newMean, std, newMax, addFactor, divideFactor]
        # indices for reference:   [      0       ,    1   ,    2   ,  3 ,    4  ,     5    ,      6      ]
        if newValue > transformation[4]:
            newValue = transformation[4]                                            # Cap newValue to newMax if it is too large
        normalized_single_score = (newValue + transformation[5])/transformation[6]  # Normalize the new score
        if transformation[6] == 0:
            return 1
        return normalized_single_score

    total_score = 0
    for i in range(len(stim_name_list)):
        curr_data_volt = data_volts_list[i]
        curr_target_volt = target_volts_list[i]
        for j in range(len(score_function_ordered_list)):
            curr_sf = score_function_ordered_list[j].decode('ascii')
            curr_weight = weights[len(score_function_ordered_list)*i + j]
            transformation = h5py.File(scores_path+stim_name_list[i]+'_scores.hdf5', 'r')['transformation_const_'+curr_sf][:]
            if curr_weight == 0:
                curr_score = 0
            else:
                curr_score = eval_function(curr_target_volt, curr_data_volt, curr_sf, dt)
            norm_score = normalize_single_score(curr_score, transformation)
            if np.isnan(norm_score):
                norm_score = 1
            total_score += norm_score * curr_weight
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
        print("Orig params:", self.orig_params)
        self.weights = opt_weight_list
        self.opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        print("Init target volts")
        self.target_volts_list = run_model(orig_params, self.opt_stim_list)

    def evaluate_with_lists(self, param_values):
        input_values = self.orig_params
        for i in range(len(param_values)):
            curr_opt_ind = self.opt_ind[i]
            input_values[curr_opt_ind] = param_values[i]
        p_objects = []
        self.opt_stim_list = self.opt_stim_list * 4 #TODO: remove later when you have actual stims, ask Minjune for them
        
        data_volts_list = []
        for i in range(len(self.opt_stim_list)):
            idx = i % 2
            if i > 2:
                p_objects[idx].wait()
                fn = vs_fn + str(i) + '.h5'
                curr_volts = nrnMreadH5(fn)
                data_volts_list.append(curr_volts)
                p_objects[idx] = run_model(idx, [])
            else:
                p_objects.append(run_model(i, []))

        #data_volts_list = run_model(input_values, self.opt_stim_list)
        print("DATA VOLTS LIST CURR SHAPE", np.array(data_volts_list).shape, \
              "DESIRED SHAPE:", (len(self.opt_stim_list),ntimestep))
        score = evaluate_score_function(self.opt_stim_list, self.target_volts_list, data_volts_list, self.weights)
        return [score]