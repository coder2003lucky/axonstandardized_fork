from json import load, dumps, dump
from csv import reader, writer
import numpy as np
from neuron import h

# DEFINE
data_dir =          '../Data'        # data directory (folder)
run_model_file =    './runModel.hoc'      

def main():
    h.load_file(run_model_file)
    
    # Initialize origin to soma
    soma = [c for c in h.allsec() if "soma" in str(c)][0]
    h.distance(sec=soma)
    
    # Find distance from each sections to soma
    distances = {str(sec):h.distance(sec(0.5)) for sec in h.allsec()}
        
    # Write distances to output file
    with open(f'{data_dir}/Distances.txt', 'w') as outfile:
        dump(distances, outfile)

if __name__ == "__main__":
    main()
