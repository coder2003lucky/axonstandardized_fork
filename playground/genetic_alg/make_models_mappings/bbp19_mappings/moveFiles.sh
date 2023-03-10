#python extractModel.py
#python extractModel_Mappings.py
mkdir pyNeuroGPU_unix2/python
cp runBBP.py pyNeuroGPU_unix2/python/
#cp -r ../../gen_alg_GPU/params pyNeuroGPU_unix2
cp -r params pyNeuroGPU_unix2/

cp -r ../../gen_alg_GPU/objectives pyNeuroGPU_unix2

# why these two?
cp ./Data/ParamTemplate.csv pyNeuroGPU_unix2/Data
cp ./Data/ParamMappings.txt pyNeuroGPU_unix2/Data
cp ./Data/64MDL.csv pyNeuroGPU_unix2/Data
cp ./Data/AllParams_reference.csv pyNeuroGPU_unix2/Data 



cp -r ../../gen_alg_GPU/stims pyNeuroGPU_unix2
cp -r ../../gen_alg_GPU/Data//target_volts_BBP19.csv pyNeuroGPU_unix2/Data
cp -r ../../gen_alg_GPU/Data/AllParams_for_params.csv pyNeuroGPU_unix2/Data
cp pyNeuroGPU_unix2/Data/Stim_raw.csv pyNeuroGPU_unix2/Data/Stim_raw.csv
cp pyNeuroGPU_unix2/Data/times.csv pyNeuroGPU_unix2/Data/times0.csv




cp extractModel_mappings.py pyNeuroGPU_unix2/python/
cp stim_converter.py pyNeuroGPU_unix2/python/

cp extractModel.py pyNeuroGPU_unix2/python/
cp file_io.py pyNeuroGPU_unix2/python/
cp auxilliary.py pyNeuroGPU_unix2/python/
cp neuron_object.py pyNeuroGPU_unix2/python/
cp proc_add_param_to_hoc_for_opt.py pyNeuroGPU_unix2/python/
cp cell.py pyNeuroGPU_unix2/python/
cp nrn_structs.py pyNeuroGPU_unix2/python/
cp create_auxilliary_data_3.py pyNeuroGPU_unix2/python/
cp make_tree_from_parent_vec.py pyNeuroGPU_unix2/python/
cp get_parent_from_neuron.py pyNeuroGPU_unix2/python/
cp runModel.hoc pyNeuroGPU_unix2/python/
cp mosinit.hoc pyNeuroGPU_unix2/python/
cp morphology.hoc pyNeuroGPU_unix2/python/
cp init.hoc pyNeuroGPU_unix2/python/
cp biophysics.hoc pyNeuroGPU_unix2/python/
cp constants.hoc pyNeuroGPU_unix2/python/
cp createsimulation.hoc pyNeuroGPU_unix2/python/
cp template.hoc pyNeuroGPU_unix2/python/
cp fitCori_bbp.hoc pyNeuroGPU_unix2/python/
cp basicCompareBBP.ipynb pyNeuroGPU_unix2/python/
# perhaps just move all .o, .c and .hoc and .mod
cp *.mod pyNeuroGPU_unix2/python/
cp *.mod2 pyNeuroGPU_unix2/python/
cp *.hoc pyNeuroGPU_unix2/python/
cp *.h pyNeuroGPU_unix2/python/

cp *.c pyNeuroGPU_unix2/python/
cp *.o pyNeuroGPU_unix2/python/


cp -r CParsed  pyNeuroGPU_unix2/python/
cp -r morphology  pyNeuroGPU_unix2/python/
cp -r params  pyNeuroGPU_unix2/python/
cp -r Stims  pyNeuroGPU_unix2/python/



cd pyNeuroGPU_unix2/src 
module load cuda
make

cd ../python/
nrnivmodl


cp ../../Data/Stim_raw1.csv ../Data/Stim_raw0.csv
python stim_converter.py


#make neuroGPU todo
