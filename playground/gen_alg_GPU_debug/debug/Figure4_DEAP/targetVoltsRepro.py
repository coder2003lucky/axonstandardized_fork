import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import os
import subprocess
import shutil
import bluepyopt as bpop
import struct
import time
import pandas as pd
import efel_ext
import time
import glob
import ctypes
import matplotlib.pyplot as plt
import bluepyopt.deapext.algorithms as algo
from extractModel_mappings import   allparams_from_mapping
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool

from joblib import Parallel, delayed, parallel_backend


model_dir = '../'
param_file ='./params/gen.csv'
data_dir = model_dir+'Data/'
params_table = data_dir + 'opt_table.csv'
run_dir = '../bin'
orig_volts_fn = data_dir + 'exp_data.csv'
vs_fn = model_dir + 'Data/VHotP'
times_file_path = model_dir + 'Data/times.csv'
nstims = 8
target_volts = np.genfromtxt(orig_volts_fn)
times =  np.cumsum(np.genfromtxt(times_file_path,delimiter=','))
nCpus =  multiprocessing.cpu_count()

if not os.path.isdir("/tmp/Data"):
    os.mkdir("/tmp/Data")


def run_model(self,stim_ind):
    print("running stim ind" + str(stim_ind))
    volts_fn = vs_fn + str(stim_ind) + '.dat'
    if os.path.exists(volts_fn):
        os.remove(volts_fn)
        #pass
    #path = "./sleep.sh"
    #p_object = subprocess.Popen(path, shell=True)
    p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)])
    return p_object

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


if __name__ == '__main__':
    data = np.genfromtxt(params_table,delimiter=',',names=True)
    pmin = data[0]
    pmax = data[1]
    params = data[2]
    params = np.array(list(params)).reshape(-1,1)
    param_values = np.repeat(params,10,axis=1).T
    allparams = allparams_from_mapping(param_values)
    p_obj = run_model(0,param_values)
    p_obj.wait()
    print("volts in Data successfully")
    

