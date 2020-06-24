import h5py
import subprocess
import pandas as pd
import csv
import numpy as np
import struct

filename = "../Data/AllParams.csv"
def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


print("running stim ind" + str(0))
p_object = subprocess.Popen(['../bin/neuroGPU',str(0)]) 
p_object.wait()
print(p_object.wait())
fn = '../Data/VHotP' + str(1) +  '.dat'    #'.h5'
curr_volts =  nrnMread(fn)
nindv = 4
Nt = int(len(curr_volts)/nindv)
shaped_volts = np.reshape(curr_volts, [nindv, Nt])
print("SHAPE:", shaped_volts.shape)
print( "NAN indx:",np.isnan(shaped_volts).any(axis=1))
print(shaped_volts[1,4995:5001])
