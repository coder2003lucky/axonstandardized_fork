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
import shutil
import struct
from extractModel_mappings_linux import   allparams_from_mapping
import bluepyopt.deapext.algorithms as algo
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing



########################################################################################
#TODO: clean up dead weight imports and fix file paths                                                         
########################################################################################

run_file = './run_model_cori.hoc'
run_volts_path = '../run_volts_bbp_full_gpu_tuned/'
paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based.csv'
orig_params = h5py.File(run_volts_path+'params/params_bbp_full_allen_gpu_tune.hdf5', 'r')['orig_full'][0]
scores_path = '../scores/'
objectives_file = h5py.File('./objectives/multi_stim_bbp_full_allen_gpu_tune_18_stims.hdf5', 'r')
opt_weight_list = objectives_file['opt_weight_list'][:]
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
stims_path = run_volts_path+'/stims/allen_data_stims_10000.hdf5'
target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
ap_tune_stim_name = '18'
ap_tune_weight = 0
params_opt_ind = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]




old_eval = algo._evaluate_invalid_fitness

model_dir = '../'
param_file ='./params/gen.csv'               #What is gen.csv? does it matter?
data_dir = model_dir+'/Data/'
params_table = data_dir + 'opt_table.csv'    #bbp template ORIG
run_dir = '../bin'
orig_volts_fn = data_dir + 'exp_data.csv' #ORIG volts
vs_fn = model_dir + 'Data/VHotP'
times_file_path = model_dir + '/Data/times.csv'
times =  np.cumsum(np.genfromtxt(times_file_path,delimiter=','))
#nCpus =  multiprocessing.cpu_count()
nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
nCpus =  multiprocessing.cpu_count()
nstims = 4

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
ntimestep = 10000

# Value of dt in miliseconds
dt = 0.02

def run_model(stim_ind, params):
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
        old_stim = "../Data/Stim_raw{}.csv"
        old_time = "../Data/times_{}.csv"
        if os.path.exists(old_stim) :
            os.remove(old_stim)
            #os.remove(old_time)
    for i in range(len(opt_stim_name_list)):
        stim = opt_stim_name_list[i].decode("utf-8")
        dt = allen_stim_file[stim+'_dt'][:]
        np.savetxt("../Data/Stim_raw{}.csv".format(i), 
                   allen_stim_file[stim][:],
                   delimiter=",")
        np.savetxt("../Data/dt_{}.csv".format(i), 
                   allen_stim_file[stim+'_dt'][:],
                   delimiter=",")
#         np.savetxt("../Data/times{}.csv".format(i), 
#                      np.array([dt for i in range(10000)]),   #TODO: how to get time so I dont get nan
#                    delimiter=",")

def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)


def stim_swap(idx, i):
    old_stim = '../Data/Stim_raw' + str(idx) + '.csv'
    old_time = '../Data/times' + str(idx) + '.csv'
    if os.path.exists(old_stim):
        os.remove(old_stim)
        #os.remove(old_time)
    os.rename(r'../Data/Stim_raw' + str(i) + '.csv', r'../Data/Stim_raw' + str(idx) + '.csv')
    #os.rename(r'../Data/times' + str(i) + '.csv', r'../Data/times' + str(idx) + '.csv')


        
def stim_reset():
    shutil.rmtree('../Data')
    copyanything('../AllenData', '../Data')

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise        
        


class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        params_ = nrnUtils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.orig_params = orig_params
        params_ = nrnUtils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.opt_ind = params_opt_ind
        params_ = [params_[i] for i in self.opt_ind]
        self.orig_params = orig_params
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval)) for name, minval, maxval in params_]
        self.weights = opt_weight_list
        self.opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        print("Init target volts")
        self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]#[target_volts_hdf5[s][:] for s in self.opt_stim_list]
        
        #print("Params to optimize:", [(name, minval, maxval) for name, minval, maxval in params_])
        self.weights = opt_weight_list
        self.opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
        self.objectives = np.repeat(bpop.objectives.Objective('Weighted score \
        functions'),2) 
        self.ap_tune_stim_name = ap_tune_stim_name
        self.ap_tune_weight = ap_tune_weight
        self.ap_tune_target = target_volts_hdf5[self.ap_tune_stim_name][:]
       


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
    
    def eval_function(self,target, data, function, dt,i):
        '''changed from hoc eval so that it returns eval for list of indvs, not just one'''
        if function in custom_score_functions:
            score = [getattr(sf, function)(target, data[i], dt) for indv in data] 
        else:
            score = sf.eval_efel(function, target, data, dt)
        score = np.reshape(score, self.nindv)
        return score
    def normalize_scores(self,curr_scores, transformation,i):
        '''changed from hoc eval so that it returns normalized score for list of indvs, not just one
        TODO: not sure what transformation[6] does but I changed return statement to fit our 
        dimensions'''
        # transformation contains: [bottomFraction, numStds, newMean, std, newMax, addFactor, divideFactor]
        # indices for reference:   [      0       ,    1   ,    2   ,  3 ,    4  ,     5    ,      6      ]
        for i in range(len(curr_scores)):
            if curr_scores[i] > transformation[4]:
                curr_scores[i] = transformation[4]        # Cap newValue to newMax if it is too large
        normalized_single_score = (curr_scores + transformation[5])/transformation[6]  # Normalize the new score
        if transformation[6] == 0:
            #return 1
            return np.ones(len(self.nindv)) #verify
        return normalized_single_score

    def test_par(self,perm):
        i = perm[0]
        j = perm[1]
        
        curr_data_volt = self.data_volts_list[i,:,:]
        curr_target_volt = self.targV[i]
        curr_sf = score_function_ordered_list[j].decode('ascii')
        curr_weight = self.weights2[len(score_function_ordered_list)*i + j]
        transformation = h5py.File(scores_path+self.opt_stim_list2[i]+'_scores.hdf5', 'r')['transformation_const_'+curr_sf][:]
        if curr_weight == 0:
            curr_scores = np.zeros(self.nindv)
        else:
            curr_scores = self.eval_function(curr_target_volt, curr_data_volt, curr_sf, dt,i)
        norm_scores = self.normalize_scores(curr_scores, transformation,i)
        for k in range(len(norm_scores)):
            if np.isnan(norm_scores[k]):
                norm_scores[k] = 1
        return np.reshape(norm_scores * curr_weight,(-1,1)) # TO DO FIX THIS LATER
    

    
    def eval_in_par(self):
        fxnsNStims = [(x,y) for x in range(nGpus) for y in range(len(score_function_ordered_list)) ] 
        with Pool(nCpus) as p:
            res = p.map(self.test_par, fxnsNStims)
        res = np.array(list(res))
        res = res[:,:,0]
        res = np.transpose(res)
        return res
            

    def evaluate_score_function(self,stim_name_list, target_volts_list, data_volts_list, weights):
        ''' This function is meant to evaluate voltage responses compared to target volts using custom and
        non-custom score functions and weights. Calculates normalized score for each stim and each score function 
        from objectives file and just stacks them in a list.
        Parameters
        -------------------- 
        stim_name_list: list of all the stims from optimisations procedure
        target_volts_list: (?? (TBD), ntimesteps) sized list of volts
        data_volts_list: (nindv, nstimesteps) sized list of voltage responses from param sets
        weights: list of weights from optimisations procedure
        iteration: varibe used to track how many times this function has been called already
        
        Return
        --------------------
        2d list of scalar scores for each parameter set w/ shape (nindv,1)
        '''
        def eval_function(target, data, function, dt):
            '''changed from hoc eval so that it returns eval for list of indvs, not just one'''
            if function in custom_score_functions:
                score = [getattr(sf, function)(target, data[i], dt) for indv in data] 
            else:
                score = sf.eval_efel(function, target, data, dt)
            score = np.reshape(score, self.nindv)
            return score
        def normalize_scores(curr_scores, transformation):
            '''changed from hoc eval so that it returns normalized score for list of indvs, not just one
            TODO: not sure what transformation[6] does but I changed return statement to fit our 
            dimensions'''
            # transformation contains: [bottomFraction, numStds, newMean, std, newMax, addFactor, divideFactor]
            # indices for reference:   [      0       ,    1   ,    2   ,  3 ,    4  ,     5    ,      6      ]
            for i in range(len(curr_scores)):
                if curr_scores[i] > transformation[4]:
                    curr_scores[i] = transformation[4]        # Cap newValue to newMax if it is too large
            normalized_single_score = (curr_scores + transformation[5])/transformation[6]  # Normalize the new score
            if transformation[6] == 0:
                #return 1
                return np.ones(len(self.nindv)) #verify
            return normalized_single_score
        
        total_scores = np.array([])
        for i in range(nGpus): 
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
                if j == 0:
                    stim_score = norm_scores * curr_weight
                    #stim_score = np.sum(norm_scores * curr_weight)
                else:
                    stim_score += norm_scores * curr_weight
                
            if i == 0:
                total_scores = np.reshape(stim_score, (-1,1)) # np.array([stim_score])
            else:
                total_scores = np.append(total_scores,np.reshape(stim_score, (-1,1)),axis=1)
        return total_scores
    
    def ap_tune(param_values, target_volts, stim_name, weight):
        stim_list = [stim_name]
        data_volt = run_model(0,[])[0] #TODO: fix this arg.
        stims_hdf5 = h5py.File(stims_path, 'r')
        dt = stims_hdf5[stim_name+'_dt']
        return tuner.fine_tune_ap(target_volts, data_volt, weight, dt)

    def evaluate_with_lists(self, param_values):
        '''This function overrides the BPOP built in function. It is currently set up to run GPU tasks for each 
        stim in chunks based on number of GPU resources then stacks these results and sends them off to be
        evaluated. It runs concurrently so that while nGpus are busy, results ready for evaluation are evaluated.
        Parameters
        -------------------- 
        param_values: Population sized list of parameter sets to be ran through neruoGPU then scored and evaluated
        
        Return
        --------------------
        2d list of scalar scores for each parameter set w/ shape (nindv,1)
        '''
        
        #stim_reset()   #TODO dont need both of these
        convert_allen_data()
        print(np.array(param_values).shape, "PRAMS")
        print(1/0)
        nindv = len(param_values)
        self.nindv = nindv
        numParams = len(param_values[0])
        # convert these param values to allParams for neuroGPU
        #print(np.array(param_values).shape,"PARAM VALUES")
        allparams = allparams_from_mapping(param_values) #allparams is not finalized
        #makeallparams()
        self.data_volts_list = np.array([])
        nstims = len(self.opt_stim_list)
        nstims = 2
        start_time_sim = time.time()
        p_objects = []
        score = []
        start_times = []
        end_times = []
        eval_times = []
        for i in range(0, nGpus):
            start_times.append(time.time())
            p_objects.append(run_model(i, []))
            
        for i in range(0,nstims):
            idx = i % (nGpus)
            p_objects[idx].wait() #wait to get volts output from previous run then read and stack
            end_times.append(time.time())
            fn = vs_fn + str(idx) +  '.h5'    #'.h5'
            curr_volts =  nrnMreadH5(fn)
            nindv = len(param_values)
            Nt = int(len(curr_volts)/nindv)
            shaped_volts = np.reshape(curr_volts, [nindv, Nt])
            if idx == 0:
                self.data_volts_list = shaped_volts #start stacking volts
            else:
                self.data_volts_list = np.append(self.data_volts_list, shaped_volts, axis = 0)
            last_batch = i >= (nstims-nGpus) #use this so neuroGPU doesn't keep running 
            if not last_batch:
                start_times.append(time.time())
                if i != idx:
                    stim_swap(idx, i)
                p_objects[idx] = run_model(idx, []) #ship off job to neuroGPU for next iter
            if idx == nGpus-1:
                self.data_volts_list = np.reshape(self.data_volts_list, (nGpus,nindv,ntimestep))
                print('SHAPE DVOLTS:', self.data_volts_list[0][0])
                print('SHAPE DVOLTS:', self.data_volts_list[1][0])

                eval_start = time.time()
                if i == nGpus-1:
                    #print(np.array(self.target_volts_list).shape, "TARTG VOLTs")
                    #print(1/0)
                    self.targV = self.target_volts_list[:nGpus]
                    #score = self.eval_in_par()
                    score = self.evaluate_score_function(self.opt_stim_list, self.targV, self.data_volts_list, self.weights)
                else:
                    self.targV = self.target_volts_list[i:i+nGpus]
                    #score = np.append(score, self.eval_in_par(), axis = 1)
                    score = np.append(score, self.evaluate_score_function(self.opt_stim_list, self.targV, self.data_volts_list, self.weights))
                eval_end = time.time()
                eval_times.append(eval_end - eval_start)
                self.data_volts_list = np.array([])
      
        print("average neuroGPU runtime: ", np.mean(np.array(end_times) - np.array(start_times)))
        print("neuroGPU runtimes: ", np.array(end_times) - np.array(start_times))
        print("evaluation took: ", eval_times)
        print("everything took: ", eval_end - start_time_sim)
        
        #ap_tune_score = ap_tune(self.params, self.ap_tune_target, self.ap_tune_stim_name, self.ap_tune_weight)
       
        #return [score + ap_tune_score]
        score = np.reshape(score,(nindv,nstims))
        score = np.reshape(np.sum(score,axis=1), (-1,1)) # sum over stims, axis =0
        # gives answer that looks right but it should be axis=1 to sum over stims
        print(score, "SCORE SHAPE")
        return score

    
algo._evaluate_invalid_fitness =hoc_evaluator.my_evaluate_invalid_fitness
