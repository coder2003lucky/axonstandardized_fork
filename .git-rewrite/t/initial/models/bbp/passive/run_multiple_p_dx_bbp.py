import os
import neuron as nrn

def load_and_run(run_file, stim_file, times_file, out_file):
	nrn.h.load_file(run_file)
	nrn.h.stimFile = stim_file
	nrn.h.timesFile = times_file
	nrn.h.outFile = out_file

	nrn.h("runModel()")

run_file = './run_model_cori_p_dx_bbp.hoc'
times = './times_0.02_2.0prepost.csv'
stims_path = './x2stims/'
volts_path = './volts/pdx/'
stims_name_list = os.listdir(stims_path)

for stim in stims_name_list:
	load_and_run(run_file, stims_path + stim, times, volts_path + 'p_dx_' + stim)
