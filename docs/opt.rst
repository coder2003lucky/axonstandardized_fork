Optimizations
====================================

The primary optimization being used is BluepyOpt's wrapper around DEAP Optimization's genetic algorithm. A very basic of the genetic algorithm being used is as follows:
1. Instantiate a population of neuron models
2. Stimulate the population and produce voltages
3. Evaluate which neurons produce the most realistic voltages to a target voltage
4. Mutate them to search the parameter space looking for the best indiviudals
5. repeat
6. stopping criteria and select parameters of the best indiviudal

The python program responsible for running this optimization is optimize_parameters_genetic_alg.py and can be found in axonstandardized/playground/gen_alg_GPU/python. To run a basic version of this optimization:
`srun python optimize_parameters_genetic_alg.py --offspring_size 2 --max_ngen 1`

.. argparse::
   :filename: ../playground/gen_alg_GPU/python/optimize_parameters_genetic_alg.py
   :func: get_parser
   :prog: optimize_parameters_genetic_alg.py




..
        TODO: 1. link bluepyopt 2. restructure axonstandardized to have neuroGPU at the top? See NeuroGPU todo

