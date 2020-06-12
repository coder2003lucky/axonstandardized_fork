# axonstandardized
Dev and Zander made some standardization updates to the axon research run procedure. Started 10/15/19.

To run neurogpu
===================================
1. make sure your account is configured for gpu access
2. run commands found in tostart_GA_GPU
3. set capacity variable in hoc_evaluator to number of gpus you allocated
4. srun



To run
=======================================
1. Fill out input.txt with params, model, peeling and number of trials to run
2. sh run.sh, leave terminal open and wait.
3. done!


Steps in run.sh
=======================================
1. Generate Params using input.txt and makeParamSet.py
    - input.txt is parsed with makeParamSet.py in param_stim_generator/make_paramset_hdf5/ folder
      - saves as h5py file on users scratch account @ h5py.File('/global/cscratch1/sd/' + user + '/run_volts/run_volts_model_peeling/params/params.hdf5', 'a')
2. Script parses input.txt to set instance variables, makes directories, and edits sbatch scripts so they match.
3. Script hits run_volts.slr which creates volts and volts_sandboxes in runs/model_peeling_date/
4. Script waits for slurm to complete and then moves slurm into runs/model_peeling_date/ (do not exit program here)
5. Script hits run_scores.slr which creates scores and score_sandbox in runs/model_peeling_date/
6. Script waits for slurm to complete and then moves slurm into runs/model_peeling_date/ (do not exit program here)
7. logs stuff to spreadsheets

initial is how we received files originally

![Proposed File Struct](/proposed_file_struct.png)


TODO: fix passive params template, it has an n/a where it shouldn't. give zander mainen specific run files?
