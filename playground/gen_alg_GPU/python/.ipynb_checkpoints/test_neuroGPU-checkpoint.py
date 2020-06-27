import unittest
from unittest.mock import patch
from hoc_evaluatorGPU_allen_par import hoc_evaluator
import numpy as np
import struct
import unittest
import sys, os
from contextlib import contextmanager
import optimize_parameters_genetic_alg
import subprocess
from extractModel_mappings_linux import   allparams_from_mapping
import multiprocessing


# Number of timesteps for the output volt.
ntimestep = 10000
model_dir = '..'
data_dir = model_dir+'/Data/'
run_dir = '../bin'
vs_fn = model_dir + '/Data/VHotP'
nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
nCpus =  multiprocessing.cpu_count()


def nrnMread(fileName):
            f = open(fileName, "rb")
            nparam = struct.unpack('i', f.read(4))[0]
            typeFlg = struct.unpack('i', f.read(4))[0]
            return np.fromfile(f,np.double)
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    res = np.fromfile(f,np.double)
    f.close      
    return res


class test_neuroGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('initializing')
        
    def setUp(self):
        self.NG = hoc_evaluator()
        self.NG.nindv = 1
        print('initialized')

    def test_initialization(self):
        """
        1. check that starting param bounds are correct
        2. check volts are shaped (stims, ntimesteps)
        """
        assert(len(self.NG.orig_params) > 12)
        for i in range(len(self.NG.orig_params)):
            assert self.NG.pmin[i] <= self.NG.pmax[i]
            assert self.NG.orig_params[i] <= self.NG.pmax[i]
            
        # check target volt shape and stim list shape
        targ_volts_shape = np.array(self.NG.target_volts_list).shape
        assert targ_volts_shape[0] == len(self.NG.opt_stim_list)
        assert targ_volts_shape[1] == ntimestep
        
    def test_run_model_dat(self):
        """runs unit test for neuroGPU src code"""
        allparams = allparams_from_mapping(np.reshape(self.NG.orig_params,(1,-1)))
        p_object = self.NG.run_model(str(0),[])
        p_object.wait()
        fn = '../Data/VHotP' + str(0) +  '.dat'    #'.h5'
        curr_volts =  nrnMread(fn)
        fn = vs_fn + str(0) +  '.dat'    #'.h5'
        assert np.isnan(curr_volts).any() ==  False
        Nt = int(len(curr_volts)/self.NG.nindv)
        self.NG.data_volts_list = np.reshape(curr_volts, [self.NG.nindv, Nt])

        
    def test_writingDTSandStims(self):
        """ tests that everything is being ran during runtime"""
        self.NG.convert_allen_data()
        for i in range(len(self.NG.opt_stim_list)):
            old_stim = "../Data/Stim_raw" + str(i) + ".csv"
            old_time = "../Data/times_"  + str(i) + ".csv"
            assert os.path.exists(old_stim)
            
    def test_eval_funct(self):
        """assert score is the right shape from eval function"""
        self.NG.convert_allen_data()
        fn = vs_fn + str(0) +  '.dat'    #'.h5'
        curr_volts =  nrnMread(fn)
        assert np.isnan(curr_volts).any() ==  False
        Nt = int(len(curr_volts)/self.NG.nindv)
        self.NG.data_volts_list = np.reshape(curr_volts, [1,self.NG.nindv, Nt])
        self.NG.data_volts_list = np.repeat(self.NG.data_volts_list,nGpus, axis =0 )
        self.NG.targV = self.NG.target_volts_list
        score = self.NG.eval_in_par()
        assert score.shape[0] == self.NG.nindv
        assert score.shape[1] == nGpus
    
    #####################################################################################
    # TODO: write some more tests to look at pkl outputs and compare params
    #####################################################################################
        
if __name__ == '__main__':
    with suppress_stdout():
        unittest.main()
    print("testing with :", " ntimesteps -", ntimestep)