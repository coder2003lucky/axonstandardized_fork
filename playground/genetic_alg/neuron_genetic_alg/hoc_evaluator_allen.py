import numpy as np
import h5py
import os
from neuron import h
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd
import math
#import ap_tuner as tuner
from config import *
from mpi4py import MPI
import allensdk.core.json_utilities as ju
# from biophys_optimize.utils import Utils
from allensdk.model.biophysical.utils import create_utils
import allensdk.model.biophysical.runner as runner
import allensdk.core.json_utilities as ju

from biophys_optimize.environment import NeuronEnvironment
import time
from sklearn.preprocessing import MinMaxScaler


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.disabled = True
# os.chdir("../../")

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

normalizers = {}
if global_rank == 0:
    for stim_name in opt_stim_name_list:
        with open(os.path.join(scores_path,'normalizers', stim_name.decode('ASCII'))+ \
                  '_normalizers.pkl','rb') as f:
            normalizers[stim_name] = pickle.load(f)
normalizers = comm.bcast(normalizers, root=0)
    


global GEN
GEN = 0
# constant to scale passsive scores by
if passive:
    PASSIVE_SCALAR = 1
else:
    PASSIVE_SCALAR = .01 # turns passive scores off

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']


def un_nest_score(score):
    res = []
    
    if score == None:
        return []
    if (type(score) == list or type(score) == np.array) and not len(score):
        return []
    
    if type(score) == np.float64:
        return [score]

    for elem in score:
        if type(elem) == np.float64:
            res.append(elem)
        else:
            res += un_nest_score(elem)
    return res

def run_model(param_set, stim_name_list):
    
    description = runner.load_description(args)
    
    
    utils = runner.create_utils(description)
    h = utils.h
   
    # configure model
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY').encode('ascii', 'ignore')
    morphology_path = morphology_path.decode("utf-8")
    utils.generate_morphology(morphology_path)
    utils.load_parameters(param_set)
    responses = []
    for sweep in stim_name_list:
        dt = stim_file[str(sweep.decode('ascii')) + "_dt"][:][0]
        stim = stim_file[str(sweep.decode('ascii')) ][:]

        sweep = int(str(sweep.decode('ascii')))
        # configure stimulus and recording
        stimulus_path = description.manifest.get_path('stimulus_path')
        run_params = description.data['runs'][0]
        # change this so they don't change our dt
        v_init = target_volts_hdf5[str(sweep)][0] # - 14
        utils.setup_iclamp2(stimulus_path, sweep=sweep, stim=stim, dt=dt, v_init=v_init)
        vec = utils.record_values()
        tstart = time.time()
        # ensure they don't change dt during the sim
        if abs(dt*h.nstep_steprun*h.steps_per_ms - 1)  != 0:
            h.steps_per_ms = 1/(dt * h.nstep_steprun)
            
        h.finitialize()
        h.run()
        tstop = time.time()
        res =  utils.get_recorded_data(vec)
        # rescale recorded data to mV
        res['v'] = res['v']*1000
        responses.append(res['v'] )

    return responses
        
    
def evaluate_score_function(stim_name_list, target_volts_list, data_volts_list, weights):
    
    def passive_chisq(target, data):
        return np.linalg.norm(target-data)**2   / np.linalg.norm(target)

    def eval_function(target, data, function, dt):
        if function in custom_score_functions:
            score = getattr(sf, function)(target, data, dt)
        else:
            score = sf.eval_efel(function, target, data, dt)
        return score
   

    total_score = 0
    psv_scores = 0
    actv_scores = 0
    active_ind = 0
    for i in range(len(stim_name_list)):
        curr_data_volt = data_volts_list[i]
        curr_target_volt = target_volts_list[i]
        stims_hdf5 = h5py.File(stims_path, 'r')
        dt_name = stim_name_list[i]
        if type(dt_name) != str:
            dt_name = dt_name.decode('ASCII')
        dt = stims_hdf5[dt_name+'_dt'][0]
        assert dt < .2
        
        curr_stim = stims_hdf5[dt_name][:]
        # HANDLE PASSIVE STIM
        if np.max(curr_target_volt) < 0:
            psv_score = PASSIVE_SCALAR * len(score_function_ordered_list) * passive_chisq(curr_target_volt, curr_data_volt)
            total_score += psv_score
            psv_scores += psv_score
            continue
        # HANDLE ACTIVE STIM
        for j in range(len(score_function_ordered_list)):
            curr_sf = score_function_ordered_list[j].decode('ascii')
            curr_weight = weights[len(score_function_ordered_list)*active_ind + j]
            
            if curr_weight == 0:
                curr_score = 0
            else:
                curr_score = eval_function(curr_target_volt, curr_data_volt, curr_sf, dt)
            if not np.isfinite(curr_score):
                norm_score = 1000 # relatively a VERY high score
            else:
                norm_score = curr_score
                norm_score = normalizers[stim_name_list[i]][curr_sf].transform(np.array(curr_score).reshape(-1,1))[0]  # load and use a saved sklean normalizer from previous step
                norm_score = min(max(norm_score,-2),2) # allow a little lee-way
                        
            # print("ACTIVE SCORE: ", norm_score * curr_weight)
            total_score += norm_score * curr_weight
            actv_scores += norm_score * curr_weight
        # we have evaled active stim, increment weight index by one
        active_ind += 1
    print('ACTIVE :', actv_scores, "PASV:", psv_scores)
    return total_score


def numpy_log(arr, base):
    if len(arr.shape) == 1:
        res = np.array([math.log(elem,base) for elem in arr])
    else:
        raise NotImplementedError
    return res


class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        super(hoc_evaluator, self).__init__()

        params = ju.read('/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/biophys_optimize/fit_styles/f9_fit_style.json')
        
        param_df = pd.read_csv('/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/params/params_allen_full.csv')
        mins, maxs = param_df['Lower bound'].values, param_df['Upper bound'].values
        
        
        self.bases = maxs / mins
        # log bound min/max
        mins, maxs = np.array([math.log(mins[i],self.bases[i]) for i in range(len(mins))]), np.array([math.log(maxs[i],self.bases[i]) for i in range(len(maxs))])
        
        fit_params = ju.read('487664663_fit.json')
        names = [elem['name'] for elem in fit_params['genome']]
        target_params = [elem['value'] for elem in fit_params['genome']]
        self.orig_params = param_df['Base value'].values
        # log bound orig params
        self.orig_params = np.array([math.log(self.orig_params[i],self.bases[i]) for i in range(len(mins))])
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval)) for name, minval, maxval in zip(names,mins,maxs)]
        print("Params to optimize:", [(name, minval, maxval) for name, minval, maxval in zip(names,mins,maxs)])
        self.weights = opt_weight_list
        self.opt_stim_list = [e for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        print("Init target volts")
        self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]
    
    
    def my_evaluate_invalid_fitness(toolbox, population):
        '''Evaluate the individuals with an invalid fitness

        Returns the count of individuals with invalid fitness
        '''
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # invalid_ind = [population[0]] + invalid_ind 
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return len(invalid_ind)


    
    def evaluate_with_lists(self, param_values):
        global GEN
        comm.Barrier() # avoid early bcast
        param_values = comm.bcast(param_values, root=0)
        score = []
        curr_rank = global_rank
        # TO DO : loop this
        while curr_rank < len(param_values):
            # second comment == test
            input_values = param_values[curr_rank] # self.orig_params#param_values[curr_rank]
            
            # undo log x-form
            input_values = np.array([math.pow(self.bases[i], input_values[i]) for i in range(len(input_values))])
            
            data_volts_list = run_model(input_values, self.opt_stim_list)
            curr_score = evaluate_score_function(self.opt_stim_list, self.target_volts_list, data_volts_list, self.weights)
            score.append(curr_score)
            curr_rank += size 
            
        comm.Barrier() # avoid early GATHER
        score = comm.gather(score, root=0)
        score = un_nest_score(score)
        score = comm.bcast(score, root=0)
        
        # score = np.concatenate([s for s in score if type(s) == np.float64 or len(s)])
        score = np.where(~np.isfinite(score), 10000000, score)
        score = np.array(score).reshape(-1,1)
        score = np.clip(score,-10000, 1200000)
        
        if global_rank  == 0:
            if GEN == 0:
                with open('score.log','w') as f:
                    f.write(str(GEN) + " : " + str(np.nanmin(score)) + ' \n')
            else:
                with open('score.log','a+') as f:
                    f.write(str(GEN) + " : " + str(np.nanmin(score)) + ' \n')
                    f.write(str(GEN) + " : len score " + str(len(score)) + ' \n')
        
        GEN += 1
        return score 



import bluepyopt.deapext.algorithms as algo
algo._evaluate_invalid_fitness = hoc_evaluator.my_evaluate_invalid_fitness

