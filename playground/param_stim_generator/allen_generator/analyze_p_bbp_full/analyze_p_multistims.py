import time
import numpy as np
import argparse
import scipy.stats as stat
import os
from noisyopt import minimizeCompass
from scipy.optimize import minimize
from heapq import heappush, heappop
import matplotlib.pyplot as plt
import operator
import pickle
from operator import itemgetter
import ipywidgets as widgets
#from IPython.display import display
#from IPython.display import clear_output
import csv
import h5py


parser = argparse.ArgumentParser(description='Analyze P Parllel')
parser.add_argument('--model', type=str, required=True, help='specifies model for AnalyzeP')

args = parser.parse_args()



save_path ='../results/{}/'.format(args.model)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
params_path = 'params/params_'
scores_path =  '../scores_14_params/'
#params_file = h5py.File(params_path, 'r')
opt_file = h5py.File("../results/{}/stims_{}.hdf5".format(args.model, args.model), 'r')
stimsInOrder = [e.decode('ascii') for e in opt_file['corresponding_original'][:]]
newStimsInOrder = [e.decode('ascii') for e in opt_file['sweep_keys'][:]]

# Constants
coarseStepSize = 0.1
fineStepSize = 0.01
obj_comb_vec = [1, 0]

# Value in range [0, 1] reflecting percent of data to use for training.
# 1 - proportionToTrain will be used for validation.
proportionToTrain = 0.7
# Integer in range [1, len(stim_names)].
# The number of top stims to use for multi-stim optimization.
# If k = 1, then use the top stim only.
k = 20#len(stimsInOrder)
# random seed to use for train/validation on optimization
seed = 500
# Weights for optimization.
param_select_vec = None
# TODO later remove hardcode
score_function_list = ['AHP1_depth_from_peak', 'AHP2_depth_from_peak', 'AHP_depth',
       'AHP_depth_abs', 'AHP_depth_abs_slow', 'AHP_depth_from_peak',
       'AHP_slow_time', 'AHP_time_from_peak', 'AP1_amp',
       'AP1_begin_voltage', 'AP1_begin_width', 'AP1_peak',
       'AP1_width', 'AP2_AP1_diff', 'AP2_AP1_peak_diff', 'AP2_amp',
       'AP2_begin_voltage', 'AP2_begin_width', 'AP2_peak',
       'AP2_width', 'AP_amplitude', 'AP_amplitude_change',
       'AP_amplitude_diff', 'AP_amplitude_from_voltagebase',
       'AP_begin_indices', 'AP_begin_time', 'AP_begin_voltage',
       'AP_begin_width', 'AP_duration', 'AP_duration_change',
       'AP_duration_half_width', 'AP_duration_half_width_change',
       'AP_end_indices', 'AP_fall_indices', 'AP_fall_rate',
       'AP_fall_rate_change', 'AP_fall_time', 'AP_height',
       'AP_rise_indices', 'AP_rise_rate', 'AP_rise_rate_change',
       'AP_rise_time', 'AP_width', 'APlast_amp', 'ISI_values',
       'KL_divergence', 'Spikecount', 'adaptation_index',
       'adaptation_index2', 'amp_drop_first_last',
       'amp_drop_first_second', 'amp_drop_second_last',
       'burst_mean_freq', 'burst_number', 'chi_square_normal',
       'depolarized_base', 'fast_AHP', 'fast_AHP_change',
       'interburst_voltage', 'irregularity_index', 'is_not_stuck',
       'isi', 'max_amp_difference', 'mean_AP_amplitude',
       'mean_frequency', 'min_voltage_between_spikes', 'peak_indices',
       'peak_time', 'peak_voltage', 'rev_dot_product',
       'spike_half_width', 'spike_width2', 'time_to_first_spike',
       'time_to_last_spike', 'time_to_second_spike', 'traj_score_1',
       'traj_score_2', 'traj_score_3', 'voltage_after_stim']

# Dump these parameters to a pickle file so that
# they can be used by the utility script file
# when it is run in the next section.
print(os.getcwd())
with open("./params.pkl", 'wb') as f:
    pickle.dump([
        scores_path,
        params_path,
        score_function_list,
        coarseStepSize,
        fineStepSize
    ], f)



# Load analysis-specific utilities from script file.
#TODO what is this?
#%run ./new_AnalyzeP.py
#placeholder as sleep, need to just do this on one node and comm info out to other nodes?
# TODO fix monkeypatch
time.sleep(5)
from new_AnalyzeP import *

ordered_stim_list, ordered_score_function_list, pin_score_dict = main()


# Fixes a shuffled order.
# Applies the same shuffled order on fed data.
# After shuffling, you can use say the first 70% for training and the last 30% for validation.
def fix_shuffled_subset_list(length):
    shuffle_pattern = np.arange(length)
    np.random.shuffle(shuffle_pattern)
    def subset_list(subset=None):
        if subset == 'train':
            return sorted(shuffle_pattern[:int(length*proportionToTrain)])
        elif subset == 'test':
            return sorted(shuffle_pattern[int(length*proportionToTrain):])
        else:
            return shuffle_pattern
    return subset_list

# Plots the sensitivity matrix as a heat map.
# Vertical axis is the parameter index and
# Horizontal axis is the numbering of each param set.
def plot_heat_map(mat, stim='', sf=''):
    # Heat Map for all elementary effects.
    #['gna_dend','gna_node','gna_soma','gkv_axon','gkv_soma','gca_dend','gkm_dend','gkca_dend','gca_soma','gkm_soma','gkca_soma','depth_cad']
    data = mat.T
    plt.imshow(data, cmap='RdBu_r', aspect='auto')
    ax = plt.gca()
    ax.invert_yaxis()
#    if stim is 'Weighted matrix':
    #plt.title('Weighted matrix', fontsize=15)
#     else:
    plt.title('Elementary effect for ' + stim + ' input\n and ' + sf + ' score function', fontsize=15)
    plt.ylabel('Parameter set', fontsize=18)
    plt.xlabel('Parameter name', fontsize=18)
    plt.colorbar()
    #plt.savefig('./heat_map_potassium.eps', format='eps', dpi=1000)
    plt.show()

def compute_lin_comb_mat(stim_name, weight_vector, sensitivity_dict):
    return sum([sensitivity_dict[stim_name][i]*weight_vector[i] for i in range(len(weight_vector))])

def plot_lin_comb_as_heat_map(stim_name, weight_vector, sensitivity_dict):
    if isinstance(stim_name, str):
        # Single stim.
        lin_comb_mat = compute_lin_comb_mat(stim_name, weight_vector, sensitivity_dict)
        print('Weighted Mat: ')
        plot_heat_map(lin_comb_mat, stim_name)
    else:
        # Multi-stim.
        lin_comb_mat = sum([
            compute_lin_comb_mat(stim_name[i], weight_vector[i*len(score_function_list):(i+1)*len(score_function_list)], sensitivity_dict) for i in range(len(stim_name))])

        plot_heat_map(lin_comb_mat, '\n & '.join(stim_name))

# @arg stim_name can either be the name of a single stim, or it can be a list of multiple stim names.
# In the latter case, optimize will be run in multi-stim mode.
def trainAndValidateScoreOptimization(stim_name, showHeatMap=False, seed=500, verbosity=True, saveToFile=False):
    np.random.seed(seed)

    if verbosity:
        plt.figure(figsize=(15,7))
        plt.title('Optimal weighted scores for ' + '\n & '.join(stim_name))
        plt.xlabel('Parameter Set Rank')
        plt.ylabel('Optimization value')

    if isinstance(stim_name, str):
        N = len(pin_score_dict[stim_name][0])
    else:
        NForEachStim = [len(pin_score_dict[single_stim][0]) for single_stim in stim_name]
        # Each of the stims in the list must be of the same length!
        assert all(N == NForEachStim[0] for N in NForEachStim), 'Each stim is not of the same length.'
        N = NForEachStim[0]
    # Create a pattern and use the pattern to select indices of training and testing data.
    shuffle_pattern = fix_shuffled_subset_list(N)
    training = shuffle_pattern('train')
    testing = shuffle_pattern('test')

    # Optimize on the training set.
    train_result, train_score_mat, _ = optimize(stim_name, training)
    train_result = train_result.x
    if verbosity:
        plt.scatter(training, train_result @ train_score_mat, label='Training Data')
        plt.plot(training, np.poly1d(np.polyfit(training, train_result @ train_score_mat, 1))(training))

    # Optimize on the entire set to establish a ground truth.
    test_result, test_score_mat, _ = optimize(stim_name)
    # Optimize on the testing set to establish a ground truth.
    #test_result, test_score_mat, _ = optimize(stim_name, testing, obj_comb_vec = obj_comb_vec)
    test_result = test_result.x
    if verbosity:
        plt.scatter(np.arange(N), train_result @ test_score_mat, label='Testing Data')
        plt.plot(np.arange(N), np.poly1d(np.polyfit(np.arange(N), train_result @ test_score_mat, 1))(np.arange(N)))
        #plt.scatter(np.arange(N), test_result @ test_score_mat, label='Ground Truth Test')
        #plt.plot(np.arange(N), np.poly1d(np.polyfit(np.arange(N), test_result @ test_score_mat, 1))(np.arange(N)))

        # Replot training and testing data on top of ground truth data.
        plt.scatter(np.arange(N), train_result @ test_score_mat, color='C1')
        plt.scatter(training, train_result @ train_score_mat, color='C0')

        # Print stims used, which were the top k stims.
        if len(stim_name) == 1:
            print('Single stim optimization:', stim_name[0])
        else:
            print('Top', min(k, len(stim_name)), 'stims:', stim_name)
        print()

        # Print weights
        print('Each row belongs to a single stim.')
        print('Training Weights:\n', train_result.reshape([min(k, len(stim_name)), len(score_function_list)]))
        print('Ground Truth Weights:\n', test_result.reshape([min(k, len(stim_name)), len(score_function_list)]))
        print()

        # Print spearman scores for the three sets of sampled data.
        print('Training Spearman:', round(stat.spearmanr(np.asarray(training), train_result @ train_score_mat)[0], 5))
        print('Testing Spearman:', round(stat.spearmanr(np.arange(N), train_result @ test_score_mat)[0], 5))
        print('Ground Truth Spearman:', round(stat.spearmanr(np.arange(N), test_result @ test_score_mat)[0], 5))

        plt.legend()
        #plt.savefig('./trainset_and_groundtruth_potassium.eps', format='eps', dpi=1000)
        plt.show()

#     if showHeatMap:
#         plot_lin_comb_as_heat_map(stim_name, train_result, sensitivity_dict)

#     if saveToFile:
#         # Save the ground truth weights reshaped into a single vector.
#         #np.savetxt("optimal_weight_list.csv", test_result.reshape([1, test_result.shape[0]]), delimiter=",")

#         # Save the stim names used in this optimization.
#         with open("optimal_stim_list.csv", 'w') as f:
#             wr = csv.writer(f, quoting=csv.QUOTE_ALL, delimiter=" ")
#             wr.writerow(stim_name)

    # Return weights
    stims_optimal_order = [np.string_(e) for e in stim_name]
    return np.array(test_result.reshape([1, test_result.shape[0]])), np.array(stims_optimal_order)

weight_list, stim_list = trainAndValidateScoreOptimization(stimsInOrder[:k], True, seed=seed, saveToFile=True)

#weight_list, stim_list = trainAndValidateScoreOptimization(newStimsInOrder[:k], True, seed=seed, saveToFile=True)

opt_result_hdf5 = h5py.File(save_path +'allen{}_objectives.hdf5'.format(args.model), 'w')
ordered_score_function_list_as_binary = np.array([np.string_(e) for e in score_function_list])
opt_result_hdf5.create_dataset('ordered_score_function_list', data=ordered_score_function_list_as_binary)
opt_result_hdf5.create_dataset('orig_opt_stim_name_list', data=stim_list)

new_opt_stim_list = np.array([np.string_(e) for e in newStimsInOrder])
opt_result_hdf5.create_dataset('opt_stim_name_list', data=new_opt_stim_list)

opt_result_hdf5.create_dataset('opt_weight_list', data=weight_list[0])
opt_result_hdf5.close()
