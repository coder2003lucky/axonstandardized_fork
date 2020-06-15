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

########################################################################################
#TODO: clean up dead weight imports and fix file paths                                                         
########################################################################################

model='bbp'
peeling='potassium'
date='04_08_2020'
params_opt_ind=[5,7,12,13,17]
run_file = './run_model_cori.hoc'
run_volts_path = '../../../run_volts/run_volts_' + model + "_" + peeling
paramsCSV = '../../playground/param_stim_generator/params_reference/params_'+ model + '_' + peeling +'.csv'
orig_params = h5py.File('params_' + model + '_' + peeling + '.hdf5', 'r')['orig_' + peeling][0]
scores_path = '../scores/'


old_eval = algo._evaluate_invalid_fitness

model_dir = '../'
param_file ='./params/gen.csv'               #What is gen.csv? does it matter?
data_dir = model_dir+'/Data/'
params_table = data_dir + 'opt_table.csv'    #bbp template ORIG
run_dir = '../bin'
orig_volts_fn = data_dir + 'exp_data.csv' #ORIG volts
vs_fn = model_dir + 'Data/VHotP'
times_file_path = model_dir + '/Data/times.csv'
nstims = 2
target_volts = np.genfromtxt(orig_volts_fn)
#target_volts = np.append(target_volts,target_volts,axis=1)

print("TARG VOLTS", target_volts.shape)
times =  np.cumsum(np.genfromtxt(times_file_path,delimiter=','))
#nCpus =  multiprocessing.cpu_count()
nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
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

allen_stim_file = h5py.File('../run_volts_bbp_full_gpu_tuned/stims/allen_data_stims_10000.hdf5', 'r')

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']

# Number of timesteps for the output volt.
ntimestep = 5000

# Value of dt in miliseconds
dt = 0.02

def run_model(stim_ind, params):
    print("running stim ind" + str(stim_ind))
    volts_fn = vs_fn + str(stim_ind) + '.h5'
    if os.path.exists(volts_fn):
        os.remove(volts_fn)
    #p_object = subprocess.Popen(['../bin/h5NeuroGPU',str(stim_ind)])
    p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)])
    return p_object

# opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
# allen_stim_file = h5py.File('../run_volts_bbp_full_gpu_tuned/stims/allen_data_stims_10000.hdf5', 'r')

# convert the allen data and save as csv
def convert_allen_data():
    for i in range(len(opt_stim_name_list)):
        stim = opt_stim_name_list[i].decode("utf-8")
        np.savetxt("../Data/stim_{}.csv".format(i), 
                   allen_stim_file[stim][:],
                   delimiter=",")
        np.savetxt("../Data/dt_{}.csv".format(i), 
                   allen_stim_file[stim+'_dt'][:],
                   delimiter=",")

def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

def makeallparams():
    filename = "../Data/AllParams.csv"
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




class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        params_ = nrnUtils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        
        self.orig_params = orig_params
        
        data = np.genfromtxt(params_table,delimiter=',',names=True)
        print("DAT SHAPE",len(data))
        self.pmin = data[0]
        print(self.pmin, "PMIN")
        self.pmax = data[1]
        self.ptarget = data[2]

        params = []
        for i in range(len(self.pmin)):
            params.append(bpop.parameters.Parameter('p' + str(i), bounds=(self.pmin[i],self.pmax[i])))

        self.params = params
        
        #print(np.array(self.orig_params), 'orig params shape')
        #print("Params to optimize:", [(name, minval, maxval) for name, minval, maxval in params_])
        self.weights = opt_weight_list
        self.opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        #print("Init target volts")
        
        
        ########################################################################################
        #TODO: need to find a way to use pipeline to generate a different target volts somehow? 
        # code left down here to calculate targ volts on the fly.... we can revist later after 
        # everything works                                                                     
        ########################################################################################
        
#       print(np.array(self.orig_params).shape)
#       allparams = allparams_from_mapping(self.orig_params) #allparams is not finalized
#       # we can still use it to generate generally the same mappings
#       makeallparams() 
#       waitfor = run_model(orig_params, self.opt_stim_list)
#       waitfor.wait()
#       fn = vs_fn + str(0) + '.h5'
#       self.target_volts = nrnMreadH5() 


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


        
    def evaluate_score_function(self,stim_name_list, target_volts_list, data_volts_list, weights):
        #print("stims", stim_name_list, "targ volt shape:", np.array(target_volts_list).shape, \
         #     "data volts", np.array(data_volts_list).shape, "WEIGHTS:", np.array(weights).shape)
        def eval_function(target, data, function, dt):
            if function in custom_score_functions:
                ########################################################################################
                #TODO: fix score function outputs to have all the params                                                   
                ######################################################################################## 
                score = [getattr(sf, function)(target, data[i], dt) for indv in data] 
                #print(np.array(score).shape,"CUSTOM SHape")
            else:
                score = sf.eval_efel(function, target, data, dt)
                #print(np.array(score).shape,"EFel SHape")
            score = np.reshape(score, self.nindv)
            #print(np.array(score).shape, "SCore shape after eval")

            return score
        def normalize_scores(curr_scores, transformation):
            # transformation contains: [bottomFraction, numStds, newMean, std, newMax, addFactor, divideFactor]
            # indices for reference:   [      0       ,    1   ,    2   ,  3 ,    4  ,     5    ,      6      ]
            for i in range(len(curr_scores)):
                if curr_scores[i] > transformation[4]:
                    curr_scores[i] = transformation[4]        # Cap newValue to newMax if it is too large
            normalized_single_score = (curr_scores + transformation[5])/transformation[6]  # Normalize the new score
            if transformation[6] == 0:
                print("WHAT TO DO HERE?")
                print(1/0)
                return 1
            return normalized_single_score #want to produce population sized normalized scores

        total_scores = np.array([])
        for i in range(len(stim_name_list)): 
            curr_data_volt = data_volts_list[i,:,:]
            curr_target_volt = target_volts_list[i]
            for j in range(len(score_function_ordered_list)):
                curr_sf = score_function_ordered_list[j].decode('ascii')
                curr_weight = weights[len(score_function_ordered_list)*i + j]
                transformation = h5py.File(scores_path+stim_name_list[i]+'_scores.hdf5', 'r')['transformation_const_'+curr_sf][:]
                if curr_weight == 0:
                    curr_scores = np.zeros(self.nindv)
                else:
                    curr_scores = eval_function(curr_target_volt, curr_data_volt, curr_sf, dt)
                norm_scores = normalize_scores(curr_scores, transformation)
                for k in range(len(norm_scores)):
                    if np.isnan(norm_scores[k]):
                        norm_scores[k] = 1
                if i == 0 and j == 0:
                    total_scores = np.reshape(norm_scores * curr_weight,(-1,1))
                else:
                    #print(total_scores.shape, "TOTAL SCORE SHAPE")
                    total_scores = np.append(total_scores,np.reshape(norm_scores * curr_weight,(-1,1)),axis=1)
        return total_scores

    def evaluate_with_lists(self, param_values):
        #param values passed in with shape (number of individuals, number of parameters)
        nindv = len(param_values)
        print(np.array(param_values).shape, "Param values")
        self.nindv = nindv
        numParams = len(param_values[0])
        # convert these param values to allParams for neuroGPU
        allparams = allparams_from_mapping(param_values) #allparams is not finalized
        makeallparams()
        p_objects = []
            
        ########################################################################################
        #TODO: get correct files configured so I dont have to fake these                                               
        ########################################################################################
        self.opt_stim_list2 = self.opt_stim_list * 2
        weights = np.repeat(self.weights,2)
        data_volts_list = np.array([])
        nstims = len(self.opt_stim_list2)
    
        
        for i in range(0, nGpus):
            p_objects.append(run_model(i, []))
        for i in range(0,nstims):
            idx = i % (nGpus)
            p_objects[idx].wait() #wait to get volts output from previous run then read and stack
            fn = vs_fn + str(idx) +  '.dat'    #'.h5'
            curr_volts = nrnMread(fn) #nrnMreadH5(fn)
            nindv = len(param_values)
            Nt = int(len(curr_volts)/nindv)
            shaped_volts = np.reshape(curr_volts, [nindv, Nt])
            if i == 0:
                data_volts_list = shaped_volts #start stacking volts
            else:
                data_volts_list = np.append(data_volts_list, shaped_volts, axis = 0)
            last_batch = i > (nstims-nGpus) #use this so neuroGPU doesn't keep running 
            if not last_batch:
                p_objects[idx] = run_model(idx, []) #ship off job to neuroGPU for next iter                
                
        data_volts_list = np.reshape(data_volts_list, (nstims,nindv,ntimestep)) 
        
        #print("DATA VOLTS LIST CURR SHAPE", np.array(data_volts_list).shape, \
        #"DESIRED SHAPE:", (len(self.opt_stim_list2),ntimestep))
        
        score = self.evaluate_score_function(self.opt_stim_list2, target_volts, data_volts_list, weights)

        #score = self.evaluate_score_function(self.opt_stim_list, target_volts, data_volts_list, self.weights)    
        ########################################################################################
        #TODO: nevals is staying constant throughout, that needs to change
        ########################################################################################
        print(np.array(score).shape, "FINAL SHape") #score shape expected to be (nindvs, number of features*nstims)
        print(score[0])
        return score

    
algo._evaluate_invalid_fitness =hoc_evaluator.my_evaluate_invalid_fitness
