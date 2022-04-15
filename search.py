import os
from tkinter.tix import Tree

from matplotlib import pyplot as plt
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
    
    alg = args.alg
    algs = ["hyperneat", "neat"]
    experimentA = Experiment(algs[0], runs)
    experimentB = Experiment(algs[1], runs)
    experiments = []
    experiments.append(experimentA)
    experiments.append(experimentB)
    for i, experiment in enumerate(experiments):
        args.alg = algs[i]
        for run in range(runs):
            try:
                experiment.current_run = run
                # plt.ion()
                if not alg or alg == "neat":
                    neat = NEAT(args.debug)
                elif alg == "hyperneat":
                    hc.apply()
                    neat = HyperNEAT(args.debug)
                    
                neat.evolve(run)
                
                experiment.record_results(neat.fitness_over_time, neat.diversity_over_time, neat.solutions_over_time, neat.species_over_time, neat.species_threshold_over_time, neat.nodes_over_time, neat.connections_over_time, neat.solution_generation, neat.species_champs_over_time, None)
                
                # if runs<2:
                    # neat.show_best()
                    # neat.show_fitness_curve()
                    # neat.show_diversity_curve()
                # plt.ioff()
            
            except KeyboardInterrupt:
                print("Stopping early...")
                if alg == "hyperneat":
                    neat.save_best_network_image(True)
                else:
                    neat.save_best_network_image()
                    
        # experiment.plot_results(True, False)
        bootst = True
        # plot fitness
        plot_mean_and_bootstrapped_ci_over_time([experiment.experiment_results for experiment in experiments], [experiment.experiment_results for experiment in experiments], [experiment.name for experiment in experiments], "Generation", "Fitness", plot_bootstrap=bootst)
        # plot diversity 
        plot_mean_and_bootstrapped_ci_over_time([experiment.diversity_results for experiment in experiments], [experiment.diversity_results for experiment in experiments], [experiment.name for experiment in experiments], "Generation", "Diversity", plot_bootstrap=bootst)
        # plot species
        plot_mean_and_bootstrapped_ci_over_time([experiment.species_results for experiment in experiments], [experiment.species_results for experiment in experiments], [experiment.name for experiment in experiments], "Generation", "N Species", plot_bootstrap=bootst)
        plot_mean_and_bootstrapped_ci_over_time([experiment.threshold_results for experiment in experiments], [experiment.threshold_results for experiment in experiments], [experiment.name for experiment in experiments], "Generation", "Species Threshold", plot_bootstrap=bootst)

        plot_mean_and_bootstrapped_ci_over_time([experiment.nodes_results for experiment in experiments], [experiment.nodes_results for experiment in experiments], [experiment.name for experiment in experiments], "Generation", "Number of Nodes", plot_bootstrap=bootst)
        plot_mean_and_bootstrapped_ci_over_time([experiment.connections_results for experiment in experiments], [experiment.connections_results for experiment in experiments], [experiment.name for experiment in experiments], "Generation", "Number of Connections", plot_bootstrap=bootst)
   
   
    # save results to file
    with open("results.txt", "w") as f:
        for experiment in experiments:
            f.write("\n")
            f.write(experiment.name + "\n")
            f.write("Fitness\n")
            f.write(str(experiment.experiment_results) + "\n")
            f.write("Diversity\n")
            f.write(str(experiment.diversity_results) + "\n")
            f.write("Species\n")
            f.write(str(experiment.species_results) + "\n")
            f.write("Species Threshold\n")
            f.write(str(experiment.threshold_results) + "\n")
            f.write("Nodes\n")
            f.write(str(experiment.nodes_results) + "\n")
            f.write("Connections\n")
            f.write(str(experiment.connections_results) + "\n")
            f.write("Solutions\n")
            f.write(str(experiment.solutions_results) + "\n")
            f.write("Species Champs\n")
            f.write(str(experiment.species_champs_results) + "\n")
            f.write("\n")
    
    plt.show()
    
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
    # parser.add_argument('-o','--obstacles', action='store_true', help='Use obstacles.')

    args = parser.parse_args() 
    main(args)    