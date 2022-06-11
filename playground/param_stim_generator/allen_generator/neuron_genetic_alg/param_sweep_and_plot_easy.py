import os
import numpy as np
import matplotlib.pyplot as plt
import struct
import h5py
import numpy as np
import pickle
import csv
import bluepyopt as bpop
import shutil, errno
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

os.chdir("neuron_files/allen/")
from neuron import h
os.chdir("../../")
from matplotlib.backends.backend_pdf import PdfPages

os.makedirs('easy_sweep', exist_ok=True)
### CHANGE THIS #############
run_file = './neuron_files/allen/run_model_cori.hoc'
paramsCSV = './params/params_bbp_full_gpu_tuned_10_based.csv'
objectives_file = h5py.File('../results/485835016/allen485835016_objectives_passive.hdf5', 'r')
stims_path = '../results/485835016/stims_485835016_passive.hdf5'
target_volts_path = '../results/485835016/target_volts_485835016_passive.hdf5'
pkl_path = None# 'best_indvs_gen_169_with_block.pkl' ## SET THIS TO NONE IF YOU DONT WANT TO USE PKL
############################


def readParamsCSV(fileName):
    fields = ['Param name', 'Base value','Lower bound', 'Upper bound']
    df = pd.read_csv(fileName,skipinitialspace=True, usecols=fields)
    paramsList = [tuple(x) for x in df.values]
    return paramsList

opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
opt_stim_name_list  = [e.decode('ASCII') for e in opt_stim_name_list]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
target_volts_hdf5 = [target_volts_hdf5[s][:] for s in opt_stim_name_list]
stim_file = h5py.File(stims_path, 'r')
param_tbl = np.array(readParamsCSV(paramsCSV))
orig_params = param_tbl[:,1].astype(np.float64)
labels = param_tbl[:,0].astype(str)
ntimestep = 10000
opt_inds = np.array([0,1,6,9,14,15,16])
sweep_inds = np.array([0])



def best_params_from_pkl(pkl_path, opt_inds, base_params):
    """
    Set base parameters for plotting based on a pkl file
    If you set pkl path to none in the header lines, this 
    will do nothing
    """
    if not pkl_path:
        return base_params
    with open(pkl_path, 'rb') as f:
        best_inds = pickle.load(f)
    best_params = base_params
    for i in range(len(base_params)):
        if i in opt_inds:
            print(np.where(opt_inds == i)[0][0])
            best_params[i] = best_inds[-1][np.where(opt_inds == i)[0][0]]
    return best_params

def scale_params(param_values, transformations):
    """
    Scale parameters based on vector of transformations.
    There is a specigic scale for e_pas that just checks those 4 values
    """
    num_params = len(param_values[0])
    flat_idx = 1
    for param_idx in range(num_params):
        if labels[param_idx] == 'e_pas_all':
            e_transforms = [-90,-80,-55,-45]
            for t_form in e_transforms:
                param_values[flat_idx,param_idx] = t_form
                flat_idx += 1
        else:
            for t_form in transformations:
                param_values[flat_idx,param_idx] = t_form * param_values[flat_idx,param_idx]
                flat_idx += 1
    return param_values
    
def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

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


def main():
    nstims = 16 # only running first 8 stims
    transformations = [.1,.5,2,10]
    color_rotation = ['blue', 'purple', 'orange', 'red']
    assert len(transformations) == len(color_rotation), "must have one color for every transformation"

    ###### Rescale orig params #################
    param_values = best_params_from_pkl(pkl_path,opt_inds, orig_params)
    param_values =  np.array(param_values).reshape(1,-1)
    param_values = np.repeat(param_values, len(param_values[0])*4+1, axis=0) # we want 4 parameter sets for each value
    # + 1 for orig params
    param_values = scale_params(param_values, transformations) # we transform theose params
    # make epas negative
    param_values[:,1] = -np.abs(param_values[:,1]) # make epas positive in case it was negative, then make it negative
    print(param_values.shape,  " : param value shape") # check shape
    print(param_values[0])
    num_params = len(param_values[0])
    
#     plt.figure()
#     zeros_params = np.zeros_like(param_values[0])
#     zeros_params[-1] = 2
#     zeros_params[-2] = .56
#     zeros_params[0] = 1.29e-06#param_values[0][0]

#     zeros_params[1] = -600#param_values[0][1]
#     zeros_params[-3] = param_values[0][-3]
#     print(zeros_params)
#     zero_result = run_model(zeros_params,[opt_stim_name_list[4]])[0]
#     plt.plot(zero_result)
#     plt.title("params are all zeros")
#     plt.savefig("zeroplot")
#     exit()
    
    for i in range(4,nstims):
        orig_volts = run_model(param_values[0], [opt_stim_name_list[i]])[0]
        total_count = 1
        pdf = PdfPages('easy_sweep/stim{}.pdf'.format(i)) # make a pdf for every stim
        for idx in range(num_params): # iterate through parameters, 15 params are hardcoded in for now... could be dynami
            curr_target_volts =  target_volts_hdf5[i]
            fig = plt.figure()
            plt.plot(curr_target_volts, color="green", label="target", alpha=.5)
            plt.plot(orig_volts, color='black', label="Base") # unmodified parameters
            for color_idx, color in enumerate(color_rotation): # for each transformation we have a parameter set to plot
                curr_volts = run_model(param_values[total_count], [opt_stim_name_list[i]])[0]
                # TODO: what is going wrong with first 10 volts?
                plt.plot(curr_volts, color=color, label=str(transformations[color_idx]), alpha=.8, linewidth=.5)
                total_count += 1
                break
                #plt.ylim(min(curr_target_volts), max(curr_target_volts)) 
            plt.legend()
            plt.title(labels[idx] + " @ " + str(param_values[0][idx] ))
            pdf.savefig(fig)
            plt.close(fig)
        pdf.close()





if __name__ == "__main__":
    main()
