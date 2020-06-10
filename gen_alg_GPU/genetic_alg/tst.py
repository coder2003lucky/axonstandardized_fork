import h5py
import subprocess
import pandas as pd
import csv


filename = "../Data/AllParams.csv"


print("running stim ind" + str(0))
p_object = subprocess.Popen(['pyNeuroGPU_unix/bin/neuroGPU',str(0)]) 
p_object.wait()
print(p_object.wait())
