import numpy as np
import h5py
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
import glob
import ctypes
import matplotlib.pyplot as plt
from extractModel_mappings_linux import   allparams_from_mapping
import bluepyopt.deapext.algorithms as algo
from concurrent.futures import ProcessPoolExecutor as Pool
import multiprocessing
import csv
import ap_tuner as tuner



#testing
model_dir = '../'
param_file ='./params/gen.csv'               #What is gen.csv? does it matter?
data_dir = model_dir+'Data/'
params_table = data_dir + 'opt_table.csv'    #bbp template ORIG

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
params_opt_ind = [0, 1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#params_opt_ind = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
model_dir = '..'
data_dir = model_dir+'/Data/'
run_dir = '../bin'
vs_fn = model_dir + '/Data/VHotP'
nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
nCpus =  multiprocessing.cpu_count()
allen_stim_file = h5py.File('../run_volts_bbp_full_gpu_tuned/stims/allen_data_stims_10000.hdf5', 'r')
old_eval = algo._evaluate_invalid_fitness

print("USING nGPUS: ", nGpus, " and USING nCPUS: ", nCpus)



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

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

def stim_swap(idx, i):
    old_stim = '../Data/Stim_raw' + str(idx) + '.csv'
    old_time = '../Data/times' + str(idx) + '.csv'  # return to this when neuroGPU reads times_i
    if os.path.exists(old_stim):
        os.remove(old_stim)
        os.remove(old_time)
    os.rename(r'../Data/Stim_raw' + str(i) + '.csv', r'../Data/Stim_raw' + str(idx) + '.csv')
    os.rename(r'../Data/times' + str(i) + '.csv', r'../Data/times' + str(idx) + '.csv')

class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        data = nrnUtils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.orig_params = orig_params
        self.opt_ind = params_opt_ind
        data = np.array([data[i] for i in self.opt_ind])
        self.orig_params = orig_params
        self.pmin = np.array((data[:,1]), dtype=np.float64)
        self.pmax = np.array((data[:,2]), dtype=np.float64) 
        self.pmax = np.delete(self.pmax, 1, 0)
        self.pmin = np.delete(self.pmin, 1, 0)
        self.ptarget = self.orig_params
        params = [] 
        for i in range(len(self.pmin)):
            params.append(bpop.parameters.Parameter(data[i][0], bounds=(self.pmin[i],self.pmax[i])))
        self.params = params
        self.weights = opt_weight_list
        self.opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
        print("Init target volts")
        self.target_volts_list = np.array([target_volts_hdf5[s][:] for s in self.opt_stim_list])
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        self.ap_tune_stim_name = ap_tune_stim_name
        self.ap_tune_weight = ap_tune_weight
        self.ap_tune_target = target_volts_hdf5[self.ap_tune_stim_name][:]
        self.dts = []
        

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
    
    def run_model(self,stim_ind, params):
        volts_fn = vs_fn + str(stim_ind) + '.dat'
        if os.path.exists(volts_fn):
            os.remove(volts_fn)
        p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)])
        return p_object
    
    # convert the allen data and save as csv
    def convert_allen_data(self):
        for i in range(len(opt_stim_name_list)):
            old_stim = "../Data/Stim_raw{}.csv"
            old_time = "../Data/times_{}.csv"
            if os.path.exists(old_stim) :
                os.remove(old_stim)
                os.remove(old_time)
        for i in range(len(opt_stim_name_list)):
            stim = opt_stim_name_list[i].decode("utf-8")
            dt = allen_stim_file[stim+'_dt'][:][0]
            self.dts.append(dt)
            f = open ("../Data/times{}.csv".format(i), 'w')
            wtr = csv.writer(f, delimiter=',', lineterminator='\n')
            current_times = [dt for i in range(ntimestep)]
            wtr.writerow(current_times)
            f.close()
            np.savetxt("../Data/Stim_raw{}.csv".format(i), 
                       allen_stim_file[stim][:],
                       delimiter=",")



        
    def eval_function(self,target, data, function, dt,i):
        '''changed from hoc eval so that it returns eval for list of indvs, not just one
        data has shape nindv,1000
        #####check passed#####
#         fig, ax = plt.subplots(5, figsize=(8,18))
#         for i in range(5):
#             ax[i].plot(np.arange(ntimestep), data[i,:])
#         plt.savefig("testing_volts.png")'''
        #print(data.shape, "VOLTS For eval, should be in indvs")
        #shape is nindv,1000
        #check passed
#         fig, ax = plt.subplots(5, figsize=(8,18))
#         for i in range(5):
#             ax[i].plot(np.arange(ntimestep), data[i,:])
#         plt.savefig("testing_volts.png")
        num_indvs = data.shape[0]
        if function in custom_score_functions:
            score = [getattr(sf, function)(target, data[i,:], dt) for i in range(num_indvs)]
        else:
            score = sf.eval_efel(function, target, data, dt)
            #testing
#             for i in range(len(score)):
#                 print(score[i]," : Efel for :", i)
#         if function == 'AP2_begin_voltage':
#             print(score, "SCORE")
#         score = np.reshape(score, self.nindv)
#         if function == 'AP2_begin_voltage':
#             print(score, "Reshaped SCORE")
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
            return np.ones(len(self.nindv)) #verify w/ Kyung
        return normalized_single_score

    def test_par(self,perm):
        """ i is the stim j is the score function to use"""
        i = perm[0]
        j = perm[1]
        curr_data_volt = self.data_volts_list[i,:,:]
        curr_target_volt = self.targV[i]
        curr_sf = score_function_ordered_list[j].decode('ascii')
        curr_weight = self.weights[len(score_function_ordered_list)*i + j]
        transformation = h5py.File(scores_path+self.opt_stim_list[i]+'_scores.hdf5', 'r')['transformation_const_'+curr_sf][:]
        if curr_weight == 0:
            curr_scores = np.zeros(self.nindv)
        else:
            curr_scores = self.eval_function(curr_target_volt, curr_data_volt, curr_sf, self.dts[i],i)
        norm_scores = self.normalize_scores(curr_scores, transformation,i)
        for k in range(len(norm_scores)):
            if np.isnan(norm_scores[k]):
                norm_scores[k] = 1
#         print("STIM #: ", i, "SCORE FUNCTION", curr_sf, "CURR WEIGHT", curr_weight, \
#              "norm scores", norm_scores)
        return norm_scores * curr_weight
    
    def top_SFs(self, run_num):
        """finds top ten scoring function indices for every stim
        TODO:write a unit test"""
        all_pairs = []
        last_stim = (run_num + 1) * nGpus
        first_stim = last_stim - nGpus
        if last_stim > 18:
            last_stim = 18
        print(first_stim,last_stim, "first and last")
        for i in range(first_stim, last_stim):#range(first_stim,last_stim):
            sf_len = len(score_function_ordered_list)
            curr_weights = self.weights[sf_len*i: sf_len*i + sf_len]
            #top_inds = sorted(range(len(curr_weights)), key=lambda i: curr_weights[i], reverse=True)[:10] #finds top ten biggest weight indices
            top_inds = np.where(curr_weights > 50)[0]
            pairs = list(zip(np.repeat(i%8,10), [ind for ind in top_inds])) #zips up indices with corresponding stim # mod 8 to make sure it is refrencing a relevant stim
            all_pairs.append(pairs)
        flat_pairs = [pair for pairs in all_pairs for pair in pairs] #flatten the list of tuples
        #print(flat_pairs)
        #print("ALL PAIRS", np.shape(np.array(all_pairs)))
        return flat_pairs

    
    def eval_in_par(self,run_num):
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
        fxnsNStims = self.top_SFs(run_num) # 52 stim-sf combinations (stim#,sf#)
        print(fxnsNStims)
        with Pool(nCpus) as p:
            res = p.map(self.test_par, fxnsNStims)
        res = np.array(list(res))
        print(res.shape, "RES SHAPE")
        print(len(score_function_ordered_list), "Score functions")
        res = res[:,:] #158,11
        prev_sf_idx = 0
        num_selected_stims = len(set([pair[0] for pair in fxnsNStims]))
        for i in range(num_selected_stims):
            num_sfs = prev_sf_idx + sum([1 for pair in fxnsNStims if pair[0]==i])
            if i == 0:
                weighted_sums = np.reshape(np.sum(res[prev_sf_idx:prev_sf_idx+num_sfs, :], axis=0),(-1,1))
            else:
                print(prev_sf_idx, "stim start idx", num_sfs, "stim end idx")
                curr_stim_sum = np.sum(res[prev_sf_idx:prev_sf_idx+num_sfs, :], axis=0)
                curr_stim_sum = np.reshape(curr_stim_sum, (-1,1))
                weighted_sums = np.append(weighted_sums, curr_stim_sum , axis = 1)
                print(curr_stim_sum.shape," : cur stim sum SHAPE      ", weighted_sums.shape, ": weighted sums shape")
            prev_sf_idx = num_sfs
        return weighted_sums
            
    def ap_tune(self,param_values, target_volts, stim_name, weight):
        '''TODO: is it possible to have AP tune preprocessed?'''
        stim_list = [stim_name]
        i  = int(stim_name) -1
        stim = opt_stim_name_list[i].decode("utf-8")
        np.savetxt("../Data/Stim_raw0.csv", 
                       allen_stim_file[stim][:],
                       delimiter=",")
        obj = self.run_model(0,[])
        obj.wait()
        data_volt = self.getVolts(0)
        data_volt = data_volt.flatten()
        stims_hdf5 = h5py.File(stims_path, 'r')
        dt = stims_hdf5[stim_name+'_dt']
        return tuner.fine_tune_ap(target_volts, data_volt, weight, dt)
    
    def getVolts(self,idx):
        '''Helper function that gets volts from data and shapes them for a given stim index'''
        #fn = vs_fn + str(idx) +  '.h5'    #'.h5' 
        #curr_volts =  nrnMreadH5(fn)
        fn = vs_fn + str(idx) +  '.dat'    #'.h5'
        curr_volts =  nrnMread(fn)
        Nt = int(len(curr_volts)/ntimestep)
        shaped_volts = np.reshape(curr_volts, [Nt,ntimestep])
        return shaped_volts

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
        
        self.dts = []
        self.convert_allen_data()
        self.nindv = len(param_values)
        full_params = np.insert(np.array(param_values), 1, np.abs(self.ptarget[1]), axis = 1)
        full_params = np.insert(np.array(param_values), 1, self.ptarget[1], axis = 1)

        allparams = allparams_from_mapping(list(full_params)) #allparams is not finalized
        self.data_volts_list = np.array([])
        nstims = len(self.opt_stim_list)
        nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
        #nstims = 4 # testing
        start_time_sim = time.time()
        p_objects = []
        score = []
        start_times = []
        end_times = []
        eval_times = []
        run_num = 0
        
        for i in range(0, nGpus):
            start_times.append(time.time())
            p_objects.append(self.run_model(i, []))
            
        for i in range(0,nstims):
            idx = i % (nGpus)
            p_objects[idx].wait() #wait to get volts output from previous run then read and stack
            end_times.append(time.time())
            shaped_volts = self.getVolts(idx)
            
            if idx == 0:
                print("APPENDED volts: ", i)
                self.data_volts_list = shaped_volts #start stacking volts
            else:
                print("APPENDED volts: ", i)
                self.data_volts_list = np.append(self.data_volts_list, shaped_volts, axis = 0) # check this
            first_batch = i < nGpus #use this so neuroGPU doesn't keep running 
            last_batch  = i == (nstims - 1)
            if not first_batch:
                start_times.append(time.time())
                if i != idx:
                    print("replaced ", idx, " with ", i)
                    stim_swap(idx, i)
                print('running: ',idx, "   real num:", i)
                p_objects[idx] = self.run_model(idx, []) #ship off job to neuroGPU for next iter
            if idx == nGpus-1:
                self.data_volts_list = np.reshape(self.data_volts_list, (nGpus,self.nindv,ntimestep))
                eval_start = time.time()
                if i == nGpus-1:
                    #need to shift over weights maybe?
                    self.targV = self.target_volts_list[:nGpus]
                    self.curr_dts = self.dts[:nGpus]
                    score = self.eval_in_par(run_num)
                else:
                    self.targV = self.target_volts_list[i-nGpus+1:i+1] # i = 15, i-nGpus+1 = 8, i+1 = 16 
                    self.curr_dts = self.dts[i-nGpus+1:i+1] # so therefore we get range(8,16) for dts and targ vs
                    score = np.append(score,self.eval_in_par(run_num),axis =1)
                    
                eval_end = time.time()
                eval_times.append(eval_end - eval_start)
                self.data_volts_list = np.array([])
                run_num += 1
                
            if last_batch: #end of batch
                for ii in range(3):
                    idx = ii % (nGpus)
                    p_objects[idx].wait() #wait to get volts output from previous run then read and stack
                    end_times.append(time.time())
                    shaped_volts = self.getVolts(idx)
                    if idx == 0:
                        self.data_volts_list = shaped_volts #start stacking volts
                    else:
                        self.data_volts_list = np.append(self.data_volts_list, shaped_volts, axis = 0) 
                    if ii == 2:
                        self.data_volts_list = np.reshape(self.data_volts_list, (3,self.nindv,ntimestep))
                        self.targV = self.target_volts_list[16:18]
                        self.curr_dts = self.dts[16:18]
                        score = np.append(score,self.eval_in_par(run_num),axis =1)


                
      
        #print("average neuroGPU runtime: ", np.mean(np.array(end_times) - np.array(start_times)))
        #print("neuroGPU runtimes: ", np.array(end_times) - np.array(start_times))
        #print("evaluation took: ", eval_times)
        print("everything took: ", eval_end - start_time_sim)
        score = np.reshape(np.sum(score,axis=1), (-1,1))
        # Minimum element indices in list 
        # Using list comprehension + min() + enumerate() 
        temp = min(score) 
        res = [i for i, j in enumerate(score) if j == temp] 
        print("The Positions of minimum element : " + str(res)) 
        #testing
#         for i in range(len(score)):
#             print(score[i], ": " + str(i))
            
        print(score.shape, "SCORE SHAPE")
        return score

    
algo._evaluate_invalid_fitness =hoc_evaluator.my_evaluate_invalid_fitness