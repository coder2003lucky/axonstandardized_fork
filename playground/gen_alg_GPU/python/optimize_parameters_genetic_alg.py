import bluepyopt as bpop
#import neurogpu_multistim_evaluator_SG as hoc_ev
import hoc_evaluatorGPU_allen_MPI as hoc_ev
#import hoc_evaluatorGPU_allen as hoc_ev
#import hoc_evaluatorGPU as hoc_ev

import bluepyopt.deapext.algorithms as algo
import bluepyopt.deapext.optimisations as opts
import deap
import deap.base
import deap.algorithms
import deap.tools

import pickle
import time
import numpy as np
from datetime import datetime
import argparse
import sys
import argparse
import textwrap
import logging
import os
from mpi4py import MPI

# set up environment variables
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()




logger = logging.getLogger()
gen_counter = 0
best_indvs = []
cp_freq = 1
old_update = algo._update_history_and_hof


import numpy


class StoppingCriteria(object):
    """Stopping Criteria class"""

    def __init__(self):
        """Constructor"""
        self.criteria_met = False
        pass

    def check(self, kwargs):
        """Check if the stopping criteria is met"""
        pass

    def reset(self):
        self.criteria_met = False

class MaxNGen(StoppingCriteria):
    """Max ngen stopping criteria class"""
    name = "Max ngen"

    def __init__(self, max_ngen):
        """Constructor"""
        super(MaxNGen, self).__init__()
        self.max_ngen = max_ngen

    def check(self, kwargs):
        """Check if the maximum number of iteration is reached"""
        gen = kwargs.get("gen")

        if gen > self.max_ngen:
            self.criteria_met = True
def _evaluate_invalid_fitness(toolbox, population):
    '''Evaluate the individuals with an invalid fitness
    Returns the count of individuals with invalid fitness
    '''
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    invalid_ind = [population[0]] + invalid_ind 
    fitnesses = toolbox.evaluate(invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return len(invalid_ind)

   

def _update_history_and_hof(halloffame, history, population):
    global gen_counter, cp_freq
    old_update(halloffame, history, population)
    best_indvs.append(halloffame[0])
    gen_counter = gen_counter+1
    print("Current generation: ", gen_counter)

    if gen_counter%cp_freq == 0:
        fn = '.pkl'
        save_logs(fn, best_indvs, population)


def _record_stats(stats, logbook, gen, population, invalid_count):
    record = stats.compile(population) if stats is not None else {}
    if global_rank != 30:
        logbook.record(gen=gen, nevals=invalid_count, **record)
    else:
     #   record = None
     logbook = None
#     record = comm.bcast(record, root=0)
    logbook = comm.bcast(logbook, root=0)
    if global_rank == 0:
        print('loggo: ', logbook, '\n')
    output = open("log.pkl", 'wb')
    pickle.dump(logbook, output)
    output.close()

def _get_offspring(parents, toolbox, cxpb, mutpb):
    '''return the offsprint, use toolbox.variate if possible'''
    if hasattr(toolbox, 'variate'):
        return toolbox.variate(parents, toolbox, cxpb, mutpb)
    return deap.algorithms.varAnd(parents, toolbox, cxpb, mutpb)


def _check_stopping_criteria(criteria, params):
    for c in criteria:
        c.check(params)
        if c.criteria_met:
            logger.info('Run stopped because of stopping criteria: ' +
                        c.name)
            return True
    else:
        return False


def MYeaAlphaMuPlusLambdaCheckpoint(
        population,
        toolbox,
        mu,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        cp_frequency=1,
        cp_filename=None,
        continue_cp=False):
    r"""This is the :math:`(~\alpha,\mu~,~\lambda)` evolutionary algorithm
    Args:
        population(list of deap Individuals)
        toolbox(deap Toolbox)
        mu(int): Total parent population size of EA
        cxpb(float): Crossover probability
        mutpb(float): Mutation probability
        ngen(int): Total number of generation to run
        stats(deap.tools.Statistics): generation of statistics
        halloffame(deap.tools.HallOfFame): hall of fame
        cp_frequency(int): generations between checkpoints
        cp_filename(string): path to checkpoint filename
        continue_cp(bool): whether to continue
    """

    if continue_cp:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(cp_filename, "rb"))
        population = cp["population"]
        parents = cp["parents"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        history = cp["history"]
        random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
        start_gen = 1
        if global_rank != 4:
            parents = population[:]
        else:
            parents = None
        #parents = comm.bcast(parents, root=0)
        #population = comm.bcast(population, root=0)
        logbook = deap.tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        history = deap.tools.History()

        # TODO this first loop should be not be repeated !
        invalid_count = _evaluate_invalid_fitness(toolbox, population)
        _update_history_and_hof(halloffame, history, population)
        _record_stats(stats, logbook, start_gen, population, invalid_count)

    stopping_criteria = [MaxNGen(ngen)]

    # Begin the generational process
    gen = start_gen + 1
    stopping_params = {"gen": gen}
    while not(_check_stopping_criteria(stopping_criteria, stopping_params)):
        if global_rank != 50: 
            offspring = _get_offspring(parents, toolbox, cxpb, mutpb)
            population = parents + offspring
        else:
            offspring = None
        #offspring = comm.bcast(offspring,root=0)
        #population = comm.bcast(population,root=0)

        invalid_count = _evaluate_invalid_fitness(toolbox, offspring)
        _update_history_and_hof(halloffame, history, population)
        _record_stats(stats, logbook, gen, population, invalid_count)
        # Select the next generation parents
        if global_rank != 90: 
            parents = toolbox.select(population, mu)
        else:
            parents = None
        #population = comm.bcaat(toolbox, root=0)
        #parents = comm.bcast(parents, root=0)

        logger.info(logbook.stream)

        if(cp_filename and cp_frequency and
           gen % cp_frequency == 0):
            cp = dict(population=population,
                      generation=gen,
                      parents=parents,
                      halloffame=halloffame,
                      history=history,
                      logbook=logbook,
                      rndstate=random.getstate())
            pickle.dump(cp, open(cp_filename, "wb"))
            logger.debug('Wrote checkpoint to %s', cp_filename)

        gen += 1
        stopping_params["gen"] = gen

    return population, halloffame, logbook, history




def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='L5PC example',
        epilog=textwrap.dedent('''\
The folling environment variables are considered:
    L5PCBENCHMARK_USEIPYP: if set, will use ipyparallel
    IPYTHON_PROFILE: if set, used as the path to the ipython profile
    BLUEPYOPT_SEED: The seed used for initial randomization
        '''))
    parser.add_argument('--start', action="store_true")
    parser.add_argument('--continu', action="store_false", default=False)
    parser.add_argument('--checkpoint', required=False, default=None,
                        help='Checkpoint pickle to avoid recalculation')
    parser.add_argument('--offspring_size', type=int, required=False, default=2,
                        help='number of individuals in offspring')
    parser.add_argument('--max_ngen', type=int, required=False, default=2,
                        help='maximum number of generations')
    parser.add_argument('--responses', required=False, default=None,
                        help='Response pickle file to avoid recalculation')
    parser.add_argument('--analyse', action="store_true")
    parser.add_argument('--compile', action="store_true")
    parser.add_argument('--hocanalyse', action="store_true")
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to use for optimization')
    parser.add_argument('--ipyparallel', action="store_true", default=False,
                        help='Use ipyparallel')
    parser.add_argument(
        '--diversity',
        help='plot the diversity of parameters from checkpoint pickle file')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose',
                        default=0, help='-v for INFO, -vv for DEBUG')

    return parser

def my_update(halloffame, history, population):
    global gen_counter, cp_freq
    old_update(halloffame, history, population)
    best_indvs.append(halloffame[0])
    gen_counter = gen_counter+1
    print("Current generation: ", gen_counter)

    if gen_counter%cp_freq == 0:
        fn = '.pkl'
        save_logs(fn, best_indvs, population)

def save_logs(fn, best_indvs, population):
    output = open("./best_indv_logs/best_indvs_gen_"+str(gen_counter)+fn, 'wb')
    pickle.dump(best_indvs, output)
    output.close()
    
def my_record_stats(stats, logbook, gen, population, invalid_count):
    '''Update the statistics with the new population'''
    #
    record = stats.compile(population) if stats is not None else {}
    if global_rank != 30:
        logbook.record(gen=gen, nevals=invalid_count, **record)
    else:
     #   record = None
     logbook = None
    record = comm.bcast(record, root=0)
    logbook = comm.bcast(logbook, root=0)
    if global_rank == 0:
        print('loggo: ', logbook, '\n')
        output = open("log.pkl", 'wb')
        pickle.dump(logbook, output)
        output.close()

def main():
    args = get_parser().parse_args()
    algo._update_history_and_hof = my_update
    algo._record_stats = my_record_stats
    algo.eaAlphaMuPlusLambdaCheckpoint = MYeaAlphaMuPlusLambdaCheckpoint

    logging.basicConfig(level=(logging.WARNING,
                                logging.INFO,
                                logging.DEBUG)[args.verbose],
                                stream=sys.stdout)
    #opt = create_optimizer(args)
    evaluator = hoc_ev.hoc_evaluator()
    seed = os.getenv('BLUEPYOPT_SEED', args.seed)
    opt = bpop.optimisations.DEAPOptimisation(
        evaluator=evaluator,
        #map_function=map_function,
        seed=seed,
        eta=20,
        mutpb=0.3,
        cxpb=0.7)
    pop, hof, log, hst = opt.run(max_ngen=args.max_ngen,
        offspring_size=args.offspring_size,
        continue_cp=args.continu,
        cp_filename=args.checkpoint,
        cp_frequency=1)
    if global_rank == 0:

        fn = time.strftime("_%d_%b_%Y")
        fn = fn + ".pkl"
        output = open("best_indvs_final"+fn, 'wb')
        pickle.dump(best_indvs, output)
        output.close()
        output = open("log"+fn, 'wb')
        pickle.dump(log, output)
        output.close()
        output = open("hst"+fn, 'wb')
        pickle.dump(hst, output)
        output.close()
        output = open("hof"+fn, 'wb')
        pickle.dump(hof, output)
        output.close()
        print ('Hall of fame: ', hof, '\n')
        print ('last log: ', log, '\n')
        print ('History: ', hst, '\n')
        print ('Best individuals: ', best_indvs, '\n')
if __name__ == '__main__':
    #if global_rank == 0:
        main()
