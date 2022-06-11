from json import load, dumps, dump
from csv import reader, writer
import numpy as np
from os.path import exists
from neuron import h
import time
from extractModel import run_extract
import importlib
import os
import math

# DEFINE
data_dir =          '../Data'                                          # data directory (folder)
params_dir =        '../params'                                        # params directory (folder)
all_params_for_params_fn = f'./{data_dir}/AllParams_for_params.csv'
params_file =       f'./{params_dir}/params_map.csv'                    # 12 parameters
#params_file =       f'./{params_dir}/params.csv'                    # 12 parameters
param_map =         f'./{data_dir}/ParamMappings.txt'               # mappings from new created in add_param_to_hoc_for_opt
template =          f'./{data_dir}/ParamTemplate.csv'  
distances_file =    f'./{data_dir}/Distances.txt'
model_data =        '../Data/64MDL.csv'                                     # the model data to be extracted and placed into allparams
corrections =       f'./{data_dir}/correction_mappings.json'        # corrections made after initial model runs
reference =         f'./{data_dir}/AllParams_reference.csv'          # reference .csv file to compare results with
all_params_fn =         f'./{data_dir}/AllParams.csv'          # 
run_model_file =    './runModel.hoc'      
                          # runModel.hoc

reversed_mappings = None
'''
categorize_key(key)
--------------------------------------------------
description:
    - determine which category each key belongs to
input[1]:
    - 'key' = key
output[1]:
    - category that 'key' belongs to
'''
def categorize_key(key):
    if 'apic' in key:
        return 'apical'
    elif 'axon' in key:
        return 'axonal'
    elif 'soma' in key:
        return 'somatic'

'''
flatten_dict(dct)
--------------------------------------------------
description:
    - flatten a dictionary of dictionaries for easier processing
input[1]:
    - 'dct' = dictionary to be flattened
output[1]:
    - 'dct' as a flattened dictionary i.e. a list
'''
def flatten_dict(dct):
    for k, v in list(dct.items()):
        if type(v) == dict:
            for key, val in list(v.items()):
                dct[f'{k}.{key}'] = val
            del dct[k]
    return dct

'''
format_value(num)
--------------------------------------------------
description:
    - converts 'num' to a floating point value
input[1]:
    - 'num' = value to be converted
output[1]:
    - 'num' formatted as a floating point value
'''
def format_value(num):
    
    if type(num) == str:
        return format_value(float(num))
    else:
        return '%.5E' % num

'''
check_allparams(allparams_file, reference_file)
--------------------------------------------------
description:
    - Test function to verify that 'allparams_file' created from our mapping/template is accurate to 'reference_file'
input[2]:
    - 'allparams_file' = allparams.csv created using our template mapping
    - 'reference_file' = original allparams.csv used as a reference
output:
    - terminal text listing mismatches between 'allparams_file' and 'reference_file'
'''
def check_allparams(allparams_file, reference_file):
    # open allparams_file as "allparams_matrix"
    with open(allparams_file, 'r') as af:
        allparams_matrix = list(reader(af))
    # open reference_file as "reference_matrix"
    with open(reference_file, 'r') as rf:
        reference_matrix = list(reader(rf))
    
    # initialize dictionary to collect errors in our mapping
    error_mapping = dict()

    # iterate over corresponding rows in allparams_file and reference_file
    for row_a, row_r in zip(allparams_matrix, reference_matrix):
        print('-----------------------------------------------')
        index = 0
        correct = 0
        wrong = 0
        # pairwise iteration over elements in the corresponding rows
        for elem_a, elem_r in zip(row_a, row_r):
            elem_r = float(elem_r)
            if float(elem_a) != float(elem_r):
                wrong += 1
                if index in reversed_mappings:
                    print(index, reversed_mappings[index], f'Got {elem_a}, but expected {elem_r}')
                    error_mapping[reversed_mappings[index]] = [index,elem_r]
                else:
                    print(index, f'Got {elem_a}, but expected {elem_r}')
                # print(f'Got {elem_a}, but expected {elem_r}')       # Value mismatch found; diagnostic print statement
                #This is since we using Ih as a default value...
                if float(elem_a) == 8 * 10 ** (-5):
                    #error_mapping[index] = float(elem_r)
                    if index in reversed_mappings:
                        print(index, reversed_mappings[index], f'Got {elem_a}, but expected {elem_r}')
                        error_mapping[reversed_mappings[index]] = [index,elem_r]
                    else:
                        print(index, f'Got {elem_a}, but expected {elem_r}')
            else:
                correct += 1
            index += 1
        print("correct: ", correct)
        print("wrong: ", wrong)
    return error_mapping

def my_assemble_allparams(param_to_allparam,allparams_file,params,param_mappings,allparams_template,distances):
    global reversed_mappings
    param_mappings_flat = flatten_dict(param_mappings.copy())
    reversed_mappings = {int(v): k for k, v in param_mappings_flat.items()}
    reversed_template = {x:i for i,x in enumerate(allparams_template) if x > 0}
    
    t_ref = []
    
    with open(allparams_file, 'r') as af:
        allparams_matrix = list(reader(af))
    ref_params = allparams_matrix[1][:-1]
    my_template = [ref_params[:] for _ in range(len(params))]
    for i in range(len(my_template)):
        for k, v in list(param_to_allparam.items()):
            for j in v:
                my_template[i][j] = params[i][int(k)]
        for k,j in reversed_template.items():
            name = reversed_mappings[k]
            if "apic" in name and "gIhbar_Ih" in name:
                secname = name.split('.')[0]+'.'+name.split('.')[1]
                distance = distances[secname]
                my_template[i][j] = (4*2.087000*math.exp((distance-0.000000)*(0.003100))*(0.000080))
                t_ref.append((distance, ref_params[j]))
    return my_template
        
    
'''
create_params_to_all_params()
--------------------------------------------------
description:
    - Mapping the indexes in the params file to allparams.csv
input[0]:
    -allparamscsv
output[1]:
    - Map from index in params file to allparams.csv
    
'''

def create_params_to_all_params(psize):
    with open(all_params_for_params_fn, 'r') as rf:
        reference_matrix = list(reader(rf))
        
    tmp = reference_matrix[1]
    params_in_all_params = [float(a) for a in tmp[:-1]]
    params_to_allparams = {}
    for i in range(psize):
        curr_indicator = 12300.0 + i
        curr_param_inds = [j for j, x in enumerate(params_in_all_params) if x == curr_indicator]
        params_to_allparams[i] = curr_param_inds
    return    params_to_allparams
    


'''
allparams_from_mapping()
--------------------------------------------------
description:
    - wrapper function for 'assemble_allparams'
input[0]:
    - n/a
output[1]:
    - 'allparams', the output from 'assemble_allparams'
'''
def allparams_from_mapping(params_input=None):
    global template, params_file, allparam_map, model_data, data_dir
    # get raw allparams template
    with open(template, 'r') as ap:
        allparams_template = list(map(lambda x: int(float(x)), list(reader(ap))[1][:-1]))
        
    with open(distances_file, 'r') as df:
        distances = load(df)
        
    # get parameters
    if params_input is None:
        with open(params_file, 'r') as p:
            params = [list(map(float, p[0].split(' '))) for p in list(reader(p))]
    else:
            params = params_input
    psize = len(params[0])
    # get all param mappings from json
    with open(param_map, 'r') as pm:
        param_mappings = load(pm)
    # get the mappings from extraneous params to allparams #WHY DO WE NEED THIS WE SHOULD NOT USE param_to_allparam
    #with open(allparam_map, 'r') as tm:
    #    param_to_allparam = load(tm)
    param_to_allparam = create_params_to_all_params(psize)

    allparams = my_assemble_allparams(param_to_allparam,reference,params,param_mappings,allparams_template,distances)


    # collect allparams using function above
    # allparams = assemble_allparams(
    #     allparams_template, 
    #     params, 
    #     param_to_allparam, 
    #     param_mappings, 
    #     model_mappings
    # )

    # write allparams to .csv file
    with open(all_params_fn, 'w', newline='') as ap:
        wr = writer(ap)
        wr.writerow([str(len(params))])
        for row in allparams:
            wr.writerow(list(map(format_value, row)))
    return allparams


def query_neuron(lst, model_file): # MOCKUP
    h.load_file(model_file)
    values = []
    aa=0
    for item in lst:
        h('aa = ' + item)
        val = float(h.aa)
        values.append(val)
        print(f"par is {item} values is {val}")
    return values


def query_neuron_2(lst, model_file): # MOCKUP
    '''
    maps names to values
    '''
    values = []
    h.load_file(model_file)
    h('aa=0')
    for item in lst:
        curr_val = h(item)
        if curr_val == 1:
            h('aa = ' + item)
            val = float(h.aa)
            values.append(val)
        else:
            values.append(-1.0)
    return values


'''
generate_mapping_file()
--------------------------------------------------
description:
    - calls extract model mapping to create relevant files.
input[0]:
    - n/a
output[1]:
    - 'allparams', the output from 'assemble_allparams'
'''
def main():
    os.system('cp ' + reference + ' ' + all_params_fn)
    time.sleep(5)
    allparams = allparams_from_mapping()
    # run_extract(True) # create params template and allparamsreference
    # check allparams 
    # get reference for correct output
    error_mapping = check_allparams(all_params_fn,reference)
    # get real values from neuron
    if len(error_mapping) > 0:
        lst = list(error_mapping.keys())
    
        # ===============================================================================================
        # neuron query, change to query_neuron, query_neuron_2, or query_neuron_3
        values = query_neuron_2(lst, run_model_file)
        # ===============================================================================================
        new_dct = dict()
        for name, value in zip(lst, values):
            [key,r_value] = error_mapping[name]
            if value == -1:
                value = r_value
            new_dct[key] = value
        with open(corrections, 'w') as jf:
            dump(new_dct, jf)
        allparams = allparams_from_mapping()
        # NO NEED TO CHECK AFTER IN REAL ONE
        check_allparams(all_params_fn, reference)


#first run this:
#run_extract(False)
#then restart kernel and run main:
#main()
#allparams_from_mapping()
if __name__ == "__main__":
    main()
