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
import nrnUtils
os.chdir("neuron_files/allen/")
from neuron import h
os.chdir("../../")
from matplotlib.backends.backend_pdf import PdfPages

#os.chdir("NeuroGPU/NeuroGPU_Base/VS/pyNeuroGPU_win2/NeuroGPU6/python/")



run_file = './neuron_files/allen/run_model_cori.hoc'
run_volts_path = './'
paramsCSV = run_volts_path+'params/params_bbp_full_gpu_tuned_10_based.csv'
# objectives_file = h5py.File('../results/485835016/allen485835016_opt.hdf5', 'r')
# stims_path = '../results/485835016/stims_485835016.hdf5'
# target_volts_path = '../results/485835016/target_volts_485835016.hdf5'
objectives_file = h5py.File('../results/485835016/allen485835016_objectives_passive.hdf5', 'r')
stims_path = '../results/485835016/stims_485835016_passive.hdf5'
target_volts_path = '../results/485835016/target_volts_485835016_passive.hdf5'


param_tbl = np.array(nrnUtils.readParamsCSV(paramsCSV))

base_params = param_tbl[:,1].astype(np.float64)
labels = param_tbl[:,0].astype(str)


with open('best_indvs_gen_169_with_block.pkl', 'rb') as f:
    best_inds = pickle.load(f)

    
orig_params =best_inds[-1]

scores_path = './scores/allen_scores/'
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
opt_stim_name_list  = [e.decode('ASCII') for e in opt_stim_name_list]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
target_volts_hdf5 = h5py.File(target_volts_path, 'r')
target_volts_hdf5 = [target_volts_hdf5[s][:] for s in opt_stim_name_list]

ap_tune_stim_name = '18'
stim_file = h5py.File(stims_path, 'r')
ntimestep = 10000
vs_fn = '/tmp/Data/VHotP'


opt_inds = np.array([0,1,6,9,14,15,16])
best_params = base_params
for i in range(len(base_params)):
    if i in opt_inds:
        print(np.where(opt_inds == i)[0][0])
        best_params[i] = orig_params[np.where(opt_inds == i)[0][0]]

# hack so this will work with less refactor though CLEAN this laster
orig_params = best_params


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise


def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)


def readParamsCSV(fileName):
    fields = ['Param name', 'Base value','Lower bound', 'Upper bound']
    df = pd.read_csv(fileName,skipinitialspace=True, usecols=fields)
    
    paramsList = [tuple(x) for x in df.values]
    return paramsList





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
        print(param_set)
        timestamps = np.array([dt for i in range(ntimestep)])
        h.curr_stim = h.Vector().from_python(curr_stim)
        h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
        #print(param_set[0])
        #print(1/0)
        h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
        h.ntimestep = ntimestep
        h.runStim()
        out = h.vecOut.to_python()    
        volts_list.append(out)
    return np.array(volts_list)


def main():
    nstims = 16 # only running first 8 stims
    transformations = [.1,.5,2,10]

    ###### TEN COPIES OF ORIG PARAMS FOR DEBUG #################
    param_values =  np.array(orig_params).reshape(1,-1)
    num_params = len(orig_params)
    param_values = np.repeat(param_values, len(param_values[0])*4+1, axis=0) 
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
                
    #param_values[flat_idx-1,param_idx] =0
    print(param_values.shape,  " : param value shape")
   
    
    color_rotation = ['blue', 'purple', 'orange', 'red']
    for i in range(nstims):
        fig, axs = plt.subplots(nrows=5,ncols=3,figsize=(12,8))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.5)
        orig_volts = run_model(param_values[0], [opt_stim_name_list[i]])
        
        total_count = 1
        for idx, ax in enumerate(axs.flatten()):

            curr_target_volts =  target_volts_hdf5[i]
            ax.plot(curr_target_volts, color="green", label="target")
            ax.plot(orig_volts, color='black', label="Base")
            for color_idx, color in enumerate(color_rotation):
                
                curr_volts = run_model(param_values[total_count], [opt_stim_name_list[i]])[0][10:]
                # TODO: what is going wrong with first 10 volts?
                import pdb; pdb.set_trace()
                ax.plot(curr_volts, color=color, label=str(transformations[color_idx]))
                total_count += 1
                ax.set_ylim(min(curr_target_volts), max(curr_target_volts))
                

            #ax.legend()
            ax.set_title(labels[idx])
        



        handles, fig_labels = ax.get_legend_handles_labels()
        fig.legend(handles, fig_labels, loc=(.02, .5))

        plt.savefig('sweep_plots/fig{}'.format(i))
        plt.close(fig)
        print(1/0)
        







if __name__ == "__main__":
    main()
#     for file in os.listdir('../Data'):
#         if 'h5' in file:
#             print("replacing: ", file)
#             os.remove('../Data/' + file)

#     for file in os.listdir('/tmp/Data'):
#         shutil.move("/tmp/Data/"+ file, "../Data/" + file)