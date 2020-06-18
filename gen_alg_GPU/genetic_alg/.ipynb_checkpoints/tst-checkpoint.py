import h5py
import subprocess
import pandas as pd
import csv
import numpy as np

filename = "../Data/AllParams.csv"
def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)

print("running stim ind" + str(0))
p_object = subprocess.Popen(['../bin/neuroGPU',str(0)]) 
p_object.wait()
print(p_object.wait())
fn = '../Data/VHotP' + str(0) +  '.h5'    #'.h5'
curr_volts =  nrnMreadH5(fn)
nindv = 11
Nt = int(len(curr_volts)/nindv)
shaped_volts = np.reshape(curr_volts, [nindv, Nt])
print("SHAPE:", shaped_volts.shape)
print( "NAN indx:",np.isnan(shaped_volts).any(axis=1))
