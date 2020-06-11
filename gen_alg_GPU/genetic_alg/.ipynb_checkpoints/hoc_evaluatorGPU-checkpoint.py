import numpy as np
import h5py
from neuron import h
import bluepyopt as bpop
import bluepyopt.deapext.algorithms as algo

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
from extractModel_mappings_linux import   allparams_from_mapping




model='bbp'
peeling='potassium'
date='04_08_2020'
params_opt_ind=[5,7,12,13,17]
run_file = './run_model_cori.hoc'
run_volts_path = '../../../run_volts/run_volts_' + model + "_" + peeling
paramsCSV = '../../playground/param_stim_generator/params_reference/params_'+ model + '_' + peeling +'.csv'
orig_params = h5py.File('params_' + model + '_' + peeling + '.hdf5', 'r')['orig_' + peeling][0]
scores_path = '../scores/'



model_dir = '../'
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

old_eval = algo._evaluate_invalid_fitness


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


def makeallparams():
    filename = "../Data/AllParams.csv"


    #apCsv =  pd.read_csv(filename)

    with open(filename) as f:
        guts = f.readlines()
        nSets = int(guts[0])
        del guts[0]
        output = [float(s) for line in guts for s in line[:-1].split(',')]



    output = np.array(output)
    output = np.reshape(output, (len(output),1))
    hf = h5py.File('../Data/AllParams.h5', 'w')
    hf.create_dataset('Data', data=output)
    hf.create_dataset('nSets', data=nSets)

    hf.close()


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
        #print(self.opt_ind, "OPT IND")
        #print(np.array(params_).shape, "Imported parameters")
        params_ = [params_[i] for i in self.opt_ind]
        self.orig_params = orig_params
        print(np.array(self.orig_params), 'orig params shape')
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval)) for name, minval, maxval in params_]
        #print(np.array(self.params))
        #print("SELF PARAMS")
        print("Params to optimize:", [(name, minval, maxval) for name, minval, maxval in params_])
        print("Orig params:", self.orig_params)
        self.weights = opt_weight_list
        self.opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        print("Init target volts")
        ##### TODO: fix this line so that target volts list actually runs neurogpu and get some list of volts for this
        #self.target_volts_list = run_model(orig_params, self.opt_stim_list)
        
        
    def my_evaluate_invalid_fitness(toolbox, population):
        '''Evaluate the individuals with an invalid fitness

        Returns the count of individuals with invalid fitness
        '''

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        invalid_ind = [population[0]] + invalid_ind
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return len(invalid_ind)

    def evaluate_with_lists(self, param_values):
        input_values = self.orig_params
        print(np.array(type(param_values[0][0])), "EVAL LIST param vals shape")
        allparams = allparams_from_mapping(param_values) #allparams is not finalized
        # we can still use it to generate generally the same mappings
        makeallparams() # TODO: this is a short term solution, find a better fix
#         for i in range(len(param_values)):
#             curr_opt_ind = self.opt_ind[i]
#             input_values[curr_opt_ind] = param_values[i]
        p_objects = []
        self.opt_stim_list = self.opt_stim_list * 4 #TODO: remove later when you have actual stims, ask Minjune for them
        
        data_volts_list = np.array([])
        
        nstims = len(self.opt_stim_list)
        capacity = 2 #set this to 8 later
        

        for i in range(nstims):
            idx = i % capacity
            if i > capacity:
                p_objects[idx].wait()
                fn = vs_fn + str(idx) + '.h5'
                curr_volts = nrnMreadH5(fn)
                if i - capacity == 1:
                    data_volts_list = curr_volts
                else:
                    data_volts_list = np.append(data_volts_list, curr_volts, axis = 1)
                    
                p_objects[idx] = run_model(idx, [])
            else:
                p_objects.append(run_model(i, []))

        #data_volts_list = run_model(input_values, self.opt_stim_list)
        print("DATA VOLTS LIST CURR SHAPE", np.array(data_volts_list).shape, \
              "DESIRED SHAPE:", (len(self.opt_stim_list),ntimestep))
        
        ######################################################
        #TODO: have to fix Data volts list and target volts  #
        #list so that evaluate score function will work      #
        ######################################################
        
        score = evaluate_score_function(self.opt_stim_list, target_volts, data_volts_list, self.weights)
        return [score]

    
algo._evaluate_invalid_fitness =hoc_evaluator.my_evaluate_invalid_fitness
