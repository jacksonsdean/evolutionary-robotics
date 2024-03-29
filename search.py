import json
import os
import time

from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange
from hyperneat import HyperNEAT
from neat import NEAT
import constants as c
import hyperneat_constants as hc
import argparse

from experiment import Experiment

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
    else:
        args.experiment_file = experiment_file # use default empty experiment

    name, controls, conditions = Experiment.load_conditions_file(experiment_file)

    experiments = [Experiment(condition, controls, args, runs) for condition in conditions]
    
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
        
    show_print = args.print or (len(experiments)<2 and runs < 2)
    print(f"Running experiment: {name}")
    for i, experiment in enumerate(experiments):
        print("_"*80)
        print(f"\tCondition {i} ({experiment.name})")
        if "alg" in experiment.condition.keys():
            alg = experiment.condition["alg"]
        else:
            alg = args.alg
        if alg == "hyperneat":
            hc.apply()
        experiment.apply_condition()
        
        experiment.setup_arrays()
        pbar = trange(runs)
        for run in pbar:
            try:
                experiment.current_run = run
                # plt.ion()
                if not alg or alg == "neat":
                    neat = NEAT(args.debug)
                elif alg == "hyperneat":
                    neat = HyperNEAT(args.debug)
                    
                neat.evolve(run, show_output=show_print)
                
                experiment.record_results(neat.fitness_over_time, neat.diversity_over_time, neat.solutions_over_time, neat.species_over_time, neat.species_threshold_over_time, neat.nodes_over_time, neat.connections_over_time, neat.solution_generation, neat.species_champs_over_time, None, neat.best_brain)
                # print(f"\tRun {run} complete with fitness {neat.get_best().fitness}")
                # if show_print:
                    # neat.show_best()
                    # neat.show_fitness_curve()
                    # neat.show_diversity_curve()
                # plt.ioff()
            
                # save results to file
                with open(results_filename, "r+") as f:
                    results = json.load(f)
                    try:
                        index = [r["name"] for r in results].index(experiment.name)
                    except ValueError as e:
                        index = -1
                    experiment.generate_results_dictionary()
                    if index>-1:
                        results[index]["num_runs"] += 1
                        for k in ["fitness_results", "diversity_results", "species_results", "threshold_results", "nodes_results", "connections_results"]:
                            results[index][k] = results[index][k] + [experiment.results[k][run]]
                        results[index]["gens_to_converge"] = results[index]["gens_to_converge"] + experiment.results["gens_to_converge"]
                        if np.max(results[index]["fitness_results"]) >  results[index]["brain"]["fitness"]:
                            results[index]["brain"] = experiment.results["brain"]
                    else:
                        results.append(experiment.results)

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
        print()
        print(f"\tCondition {i} complete with fitness {np.mean(experiment.fitness_results[:,-1])}\n")
    
    time.sleep(1)
    os.system(f"python ./analyze_experiment.py -f {results_filename} &")
    time.sleep(1)

    # show the best from all experimental runs
    if args.print:
        for result in results:
            robot = f"brain_best_{result['name']}"
            name = f"{robot}.nndf"
            with open(name, "w") as f:
                string = result["brain"]["network"]
                f.write(string)
                f.close()
            os.system(f"python simulate.py --body quadruped.urdf --brain {name} --best")
            time.sleep(1)
            os.system(f"python footprint_diagram.py -r {robot}")
            time.sleep(1)
  
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Run search on the robot controller.')
    parser.add_argument('-d','--debug', action='store_true', help='Show debug messages.')
    parser.add_argument('-pr','--print', action='store_true', help='Show print messages for each gen.')
    parser.add_argument('-g','--generate', action='store_true', help='Generate new world first.')
    parser.add_argument('-t','--generations', action='store', help='Number of generations to run.')
    parser.add_argument('-p','--pop', action='store', help='Population size.')
    parser.add_argument('-s','--species', action='store', help='Number of species.')
    parser.add_argument('-a','--alg', action='store', help='Algorithm to use.')
    parser.add_argument('-r','--experiment_runs', action='store', help='Number of experiment runs.')
    parser.add_argument('-b','--do_bootstrap', action='store_true', help='Show bootstrap CI on experiment plots.')
    parser.add_argument('-f','--experiment_file', action='store', help='Experiment description file.')
    # parser.add_argument('-o','--obstacles', action='store_true', help='Use obstacles.')

    args = parser.parse_args() 
    main(args)    