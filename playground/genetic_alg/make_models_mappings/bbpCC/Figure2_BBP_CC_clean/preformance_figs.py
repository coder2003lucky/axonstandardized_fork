import neuron as nrn
import os
import struct
import numpy
import subprocess
import shutil
import time

from NeuroGPUFromPkl import run_params_with_pkl
nrnTypes = {
    2: numpy.short,
    3: numpy.float32,
    4: numpy.double,
    5: numpy.int,
}
global nparams
nparams = 3

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return numpy.fromfile(f,nrnTypes[typeFlg])
def nrnMwrite(fileName,params):
    #nrn.h.fout.wopen(fileName)
    for i in range(0,(params.size/nparams)):
        vvec = nrn.h.Vector(params[i*nparams:(i+1)*nparams])
        nrn.h.writeVector(fileName,vvec)

    #nrn.h.fout.close()
def writeParamsFiles():
    params_file = './params/001_1200s1_pin.dat'
    raw_params = nrnMread(params_file)
    aa = raw_params.tolist()
    aa = aa+aa + aa+aa+aa+aa+aa+aa+aa+aa+aa+aa + aa+aa+aa+aa+aa+aa+aa+aa+aa+aa + aa+aa+aa+aa+aa+aa+aa+aa+aa + aa+aa+aa+aa+aa+aa+aa+aa+aa+aa + aa+aa+aa+aa+aa+aa+aa+aa+aa+aa + aa+aa+aa+aa+aa+aa+aa+aa+aa+aa + aa+aa+aa+aa+aa+aa+aa+aa+aa+aa + aa+aa+aa+aa+aa+aa+aa+aa+aa+aa + aa+aa+aa+aa+aa+aa+aa+aa+aa
    raw_params = numpy.array(aa)
    for i in range(0, 16):

        psize = 2 ** i
        curr_parmas = raw_params[:psize*nparams]
        paramFN = './params/figs' + str(psize) + '.dat'
        nrnMwrite(paramFN,curr_parmas)

def writeOrigParamsFiles():
        params_file = './params/orig.dat'
        raw_params = nrnMread(params_file)
        aa = raw_params.tolist()
        aa = aa + aa + aa + aa
        raw_params = numpy.array(aa)
        for i in range(0, 7):
            psize = 2 ** i
            curr_parmas = raw_params[:psize * nparams]
            paramFN = './params/figsorigs' + str(psize) + '.dat'
            nrnMwrite(paramFN, curr_parmas)



modelFile = "./runModel.hoc"
nrn.h.load_file(1, modelFile)
data_dir = 'C:/NoMech/Data/'
hocmodel_name = data_dir + os.path.basename(nrn.h.modelFile)[:-3] + 'pkl'
run_dir = 'C:/pyNeuroGPU_win2'
writeParamsFiles()
#writeOrigParamsFiles()
times = range(17)

#for i in range(0, 15):
f = open(data_dir + 'lista.csv', 'w')
for i in range(0,16):

    os.chdir('C:/NoMech/')
    psize = 2 ** i
    #run_params_with_pkl(hocmodel_name,'./params/figs' + str(psize) +'.dat',psize)
    run_params_with_pkl(hocmodel_name, './params/figs' + str(psize) + '.dat', psize)

    shutil.copy(data_dir+'AllParams.csv',run_dir +"/Data/")
    time.sleep(10)
    os.chdir(run_dir+ '/NeuroGPU6/')
    subprocess.call('NeuroGPU6.exe')
    time.sleep(10)
    my_data = numpy.genfromtxt(run_dir+ '/Data/RunTimes.csv', delimiter=',')
    print my_data[1]
    times[i] = my_data[1]
    f.write(str(times[i]) + "\n")
f.close()


