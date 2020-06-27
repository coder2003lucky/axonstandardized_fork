# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:30:07 2019

@author: bensr

"""
import pickle
import matplotlib.pyplot as plt
from extractModel_mappings_linux import   allparams_from_mapping
import os
import subprocess
import shutil
import time
from deap import base, creator, tools, algorithms
import numpy as np
import struct
import bluepyopt as bpop
import sys
import pickle
from efel_ext import eval
from collections import Counter

feature_list = ['voltage_base','steady_state_voltage_stimend','decay_time_constant_after_stim','sag_amplitude','ohmic_input_resistance','voltage_after_stim']
feature_list =  ['voltage_base','AP_amplitude','voltage_after_stim','peak_time','spike_half_width','AHP_depth','chi']
creator.create("FitnessMax", base.Fitness, weights=(-1.0,-1.0,-1.0,-1.0,-1.0,-1.0)) #-1.0,-1.0,-1.0
creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0,-1.0,-1.0,-1.0,-1.0)) #-1.0,-1.0,-1.0

#creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


model_dir = '../'

data_dir = model_dir+'/Data/'
params_table = data_dir + 'opt_table.csv'
run_dir = '../bin'
orig_volts_fn = data_dir + './exp_data.csv'
vs_fn = model_dir + 'Data/VHotP'
times_file_path = model_dir + 'Data/times.csv'
nstims = 8
ga_res_fn = model_dir + '/volts/ga_res.txt'

stim_ind = 7
NPARAMS = 14
nindvs = 1000
orig_volts = np.genfromtxt(orig_volts_fn)[stim_ind,:]
def init_nrngpu():
    global pmin
    global pmax
    global ptarget
    data = np.genfromtxt(params_table,delimiter=',',names=True)
    pmin = data[0]
    pmax = data[1]
    ptarget = data[2]
init_nrngpu()

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)
def nrnMreadOrig(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.float32)


def make_all_params_from_hof():
    fn = './hof.pkl'
    
    f = open(fn, 'rb') 
    best_indvs = pickle.load(f)
    param_mat = np.array(best_indvs)
    param_mat = param_mat[0:nindvs,:]
    np.savetxt('../Data/best_indvs.csv',param_mat)
    
    allparams = allparams_from_mapping(param_mat)
def create_volts(stims):
    all_volts = []
    for stim_ind in stims:
        volts_fn = vs_fn + str(stim_ind) + '.dat'
        if os.path.exists(volts_fn):
            os.remove(volts_fn)
        p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)])
        p_object.wait()
        curr_volts = nrnMread(volts_fn)
        Nt = int(len(curr_volts)/nindvs)
        shaped_volts = np.reshape(curr_volts, [nindvs, Nt])
        all_volts.append(shaped_volts)
    pickle.dump( all_volts, open( '../Plots/allHofVolts.pkl', "wb" ) )
    return all_volts

def get_efels(volts):
    times = np.linspace(0,999.9,5000)
    all_origs = np.genfromtxt(orig_volts_fn)
    stim_ind =0
    best_volts = []
    for curr_volts in volts:
        orig_volts = all_origs[stim_ind,:]
        
        print(f'origs length is {len(orig_volts)} is {orig_volts}\n all volts length is {len(volts)} curr volts length is  {len(curr_volts)}\n')
        
        stim_ind+=1
        
        efels = eval([orig_volts], curr_volts,times)
        mins_inds = np.argmin(efels,0)
        print (mins_inds)
        curr_feature = 0
        fig,axs = plt.subplots(3,3,figsize=(9,9))
        for i,curr_ax in zip(mins_inds,axs.flat):
            curr_str =['best ind is ',str(i),'feature is ', feature_list[curr_feature]] 
            curr_feature = curr_feature + 1
            curr_volt = curr_volts[i]
            curr_ax.plot(times,orig_volts,'r',times,curr_volt,'b')
            curr_ax.set_title(curr_str)
            
        plt.show()    
        fig.savefig('../Plots/best_efels_stim_' + str(i) + '.eps')
        fig.savefig('../Plots/best_efels_stim_' + str(i) + '.pdf')
        fig.savefig('../Plots/best_efels_stim_' + str(i) + '.png')   
    return efels

def plot_all_best_inds():
    all_volts = pickle.load( open(  '../Plots/allHofVolts.pkl', "rb" ) )
    all_volts = np.array(all_volts)
    print(np.shape(all_volts))
    times = np.linspace(0,999.9,5000)
    all_origs = np.genfromtxt(orig_volts_fn)
    stim_ind =0
    best_volts = []
    all_mins_inds = []
    try:
        all_mins = pickle.load( open(  '../Plots/allMins.pkl', "rb" ) )
        print(f'loaded all_mins from pkl {all_mins}')
    except:
        all_mins = []
    if all_mins is []:
        for curr_volts in all_volts:
            orig_volts = all_origs[stim_ind,:]
            stim_ind+=1
            efels = eval([orig_volts], curr_volts,times)
            all_mins_inds.append(np.argmin(efels,0))
            #print (mins_inds)
            curr_feature = 0
        all_mins = [item for sublist in all_mins_inds for item in sublist]
        print(f'in efelcreate allmins is {all_mins}')
        pickle.dump( all_mins, open( '../Plots/allMins.pkl', "wb" ) )
    mins_dict = Counter(all_mins)
    print(mins_dict)
    sorted_mins = {k: v for k, v in sorted(mins_dict.items(), key=lambda item: item[1],reverse=True)}
    fig,axs = plt.subplots(len(sorted_mins),nstims,figsize=(90,90))
    for (curr_min,axs_row) in zip(sorted_mins,axs):
        curr_str =['best ind is ',curr_min,'appeared ', sorted_mins[curr_min]] 
        print(f'curr_min is {curr_min} volts shape is {np.shape(all_volts)}')
        
        min_volts = all_volts[:,curr_min,:]
        for (curr_volts,curr_ax,curr_orig) in zip(min_volts,axs_row,all_origs):
            curr_ax.plot(times,curr_orig,'r',times,curr_volts,'b')
        curr_ax.set_title(curr_str)
    #fig.savefig('../Plots/all_inds.eps')
    #fig.savefig('../Plots/all_inds.pdf')
    fig.savefig('../Plots/all_inds.png')   
    
    
    


def similiar(indv1,indv2):
    ans = 0
    for ind in range(len(indv1)):
        ans = ans + abs(indv1[ind] - indv2[ind])
    if(ans<0.01):
        return True
    return False

        
def print_best_volts():
    all_volts = get_hof()
    print(all_volts)
    #all_volts = np.genfromtxt(ga_res_fn,delimiter=',')
    time = np.linspace(0,999.9,5000)
    for curr_volt in all_volts:
        #plt.plot(time,orig_volts,'r',time,curr_volt[:-1],'b')
        plt.plot(time,orig_volts,'r',time,curr_volt,'b')
        plt.show()
def print_allvolts_ind(ind_to_plot):
    all_volts = pickle.load( open(  '../Plots/allHofVolts.pkl', "rb" ) )
    times = np.linspace(0,999.9,5000)
    fig,axs = plt.subplots(3,3,figsize=(9,9))
    orig_volts = np.genfromtxt(orig_volts_fn)
    for volts,curr_ax,curr_orig in zip(all_volts,axs.flat,orig_volts):
        curr_volts = volts[ind_to_plot]
        curr_ax.plot(times,curr_orig,'r',times,curr_volts,'b')
    fig.savefig(f'../Plots/best_overall{ind_to_plot}.eps')
    fig.savefig(f'../Plots/best_overall{ind_to_plot}.pdf')
    fig.savefig(f'../Plots/best_overall{ind_to_plot}.png')

    
def main():
   # if len(sys.argv) is 1:
   #     all_params = make_all_params_from_hof()
   #     all_volts = create_volts(list(range(8)))
   #     efels = get_efels(all_volts)
   # else:
   #     ind_to_plot = int(sys.argv[1])
   #     print_allvolts_ind(ind_to_plot)
    all_params = make_all_params_from_hof()
    #all_volts = create_volts(list(range(8)))
    #plot_all_best_inds()
    #ind_to_plot = int(sys.argv[1])
    
    #print_allvolts_ind(534)
    #print_allvolts_ind(697)
    #print_allvolts_ind(86)
main()
#print_best_volts()
