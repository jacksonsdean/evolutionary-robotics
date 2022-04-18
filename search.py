import json
import os
from tkinter.tix import Tree

from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange
from hyperneat import HyperNEAT
from neat import NEAT
import constants as c
import hyperneat_constants as hc
import argparse

from experiment import Experiment
from util import plot_mean_and_bootstrapped_ci_over_time

def main(args = None):

    if args.generations:
        c.num_gens = int(args.generations)

    if args.pop:
        c.pop_size = int(args.pop)
    if args.species:
        c.species_target = int(args.species)


    if args.generate:
        os.system("python generate.py")

    if args.experiment_runs:
        runs = int(args.experiment_runs)
    else:
        runs = 1

    c.apply_condition("alg", args.alg)

    experiment_file = "experiments/empty_experiment.json"
    if args.experiment_file:
        experiment_file = args.experiment_file

    name, conditions = Experiment.load_conditions_file(experiment_file)

    experiments = [Experiment(condition, args, runs) for condition in conditions]
    
    results_filename = f"{experiment_file.split('.')[0]}_results.json"
    
    results =[]
    if os.path.exists(results_filename):
        with open(results_filename, "r") as f:
            try:
                results = json.load(f)
            except:
                ...
            
    if not os.path.exists(results_filename) or len(results)==0:
        with open(results_filename, "w+") as f:
            f.write("[\n")
            for experiment in experiments:
                experiment.generate_empty_results_dictionary()
                r = experiment.results
                r["num_runs"] = 0
                json.dump(r, f, indent=4)
                if experiment != experiments[-1]:f.write(",\n")
                
            f.write("\n]")
            f.close()
        
    
    print(f"Running experiment: {name}")
    for i, experiment in enumerate(experiments):
        print(f"\tCondition {i} ({experiment.name})")
        experiment.apply_condition()
        alg = c.alg
       
        if alg == "hyperneat":
            hc.apply()
            experiment.apply_condition()
        pbar = trange(runs)
        for run in pbar:
            try:
                experiment.current_run = run
                # plt.ion()
                if not alg or alg == "neat":
                    neat = NEAT(args.debug)
                elif alg == "hyperneat":
                    neat = HyperNEAT(args.debug)
                    
                neat.evolve(run, show_output=len(experiments)<2)
                
                experiment.record_results(neat.fitness_over_time, neat.diversity_over_time, neat.solutions_over_time, neat.species_over_time, neat.species_threshold_over_time, neat.nodes_over_time, neat.connections_over_time, neat.solution_generation, neat.species_champs_over_time, None)
                # print(f"\tRun {run} complete with fitness {neat.get_best().fitness}")
                if runs<2:
                    neat.show_best()
                    # neat.show_fitness_curve()
                    # neat.show_diversity_curve()
                # plt.ioff()
            
                # save results to file
                with open(results_filename, "r+") as f:
                    results = json.load(f)
                    index = [r["name"] for r in results].index(experiment.name)
                    if index>-1:
                        results[index]["num_runs"] += 1
                        experiment.generate_results_dictionary()
                        for k in ["fitness_results", "diversity_results", "species_results", "threshold_results", "nodes_results", "connections_results"]:
                            results[index][k] = results[index][k] + [experiment.results[k][run]]
                        results[index]["gens_to_converge"] = results[index]["gens_to_converge"] + experiment.results["gens_to_converge"]
                    f.seek(0)
                    f.truncate()
                    f.write("[\n")
                    for r in results:
                        json.dump(r, f, indent=4)
                        
                        if r != results[-1]:f.write(",\n")
                    f.write("\n]")
                    f.close()
                    
            except KeyboardInterrupt:
                print("Stopping early...")
                if alg == "hyperneat":
                    neat.save_best_network_image(True)
                else:
                    neat.save_best_network_image()
        print(f"\tCondition {i} complete with fitness {np.mean(experiment.fitness_results[:,-1])}\n")
    
    
   
  
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Run search on the robot controller.')
    parser.add_argument('-d','--debug', action='store_true', help='Show debug messages.')
    parser.add_argument('-g','--generate', action='store_true', help='Generate new world first.')
    parser.add_argument('-t','--generations', action='store', help='Number of generations to run.')
    parser.add_argument('-p','--pop', action='store', help='Population size.')
    parser.add_argument('-s','--species', action='store', help='Number of species.')
    parser.add_argument('-a','--alg', action='store', help='Algorithm to use.')
    parser.add_argument('-e','--experiment_runs', action='store', help='Number of experiment runs.')
    parser.add_argument('-b','--do_bootstrap', action='store_true', help='Show bootstrap CI on experiment plots.')
    parser.add_argument('-f','--experiment_file', action='store', help='Experiment description file.')
    # parser.add_argument('-o','--obstacles', action='store_true', help='Use obstacles.')

    args = parser.parse_args() 
    main(args)    