#python extractModel.py
#python extractModel_Mappings.py
mkdir pyNeuroGPU_win2/python
cp runBBP.py pyNeuroGPU_win2/python/
#cp -r ../gen_alg_GPU_debug/params pyNeuroGPU_win2
cp -r params pyNeuroGPU_win2/

cp -r ../gen_alg_GPU_debug/objectives pyNeuroGPU_win2

# why these two?
cp ./Data/ParamTemplate.csv pyNeuroGPU_win2/Data
cp ./Data/ParamMappings.txt pyNeuroGPU_win2/Data
cp ./Data/64MDL.csv pyNeuroGPU_win2/Data
cp ./Data/AllParams_reference.csv pyNeuroGPU_win2/Data 



cp -r ../gen_alg_GPU_debug/stims pyNeuroGPU_win2
cp -r ../gen_alg_GPU_debug/Data/target_volts_BBP19.csv pyNeuroGPU_win2/Data
cp -r ../gen_alg_GPU_debug/Data/AllParams_for_params.csv pyNeuroGPU_win2/Data



cp extractModel_mappings.py pyNeuroGPU_win2/python/
cp extractModel.py pyNeuroGPU_win2/python/
cp file_io.py pyNeuroGPU_win2/python/
cp auxilliary.py pyNeuroGPU_win2/python/
cp neuron_object.py pyNeuroGPU_win2/python/
cp proc_add_param_to_hoc_for_opt.py pyNeuroGPU_win2/python/
cp cell.py pyNeuroGPU_win2/python/
cp nrn_structs.py pyNeuroGPU_win2/python/
cp create_auxilliary_data_3.py pyNeuroGPU_win2/python/
cp make_tree_from_parent_vec.py pyNeuroGPU_win2/python/
cp get_parent_from_neuron.py pyNeuroGPU_win2/python/
cp runModel.hoc pyNeuroGPU_win2/python/