import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from util import plot_mean_and_bootstrapped_ci_over_time

# %%
def main(args):
    plt.rc('font', size=20) #controls default text size
    filename = ""
    if args.experiment_file:
        filename = args.experiment_file
    else:
        raise Exception("No experiment file specified")
    with open(filename) as f:
        data = json.load(f)

    if args:
        bootst = args.do_bootstrap

    keys = ["fitness_results",
            "diversity_results",
            "species_results",
            "threshold_results",
            "nodes_results",
            "connections_results"]
    brain = None
    experiment_name = filename.split("/")[-1].split(".")[0]
    if "args" in data[0].keys():
        experiment_config = data[0]["args"]["experiment_file"]
        with open(experiment_config) as f:
            config = json.load(f)
            experiment_name = config["name"]

    for i, d in enumerate(data):
        for k in keys:
            if len(d[k]) == 0:
                continue
            lengths = [len(d[k][j]) for j in range(len(d[k]))]
            max = np.max(lengths)
            for j, run in enumerate(d[k]):
                orig_len = len(run)
                if args.gens:
                    if orig_len > int(args.gens):
                        d[k][j] = run[:int(args.gens)]
                        print(f"Truncated {k} from {orig_len} to {len(d[k][j])}")
                    else: 
                        d[k][j] = np.array([np.nan] * int(args.gens))
                elif orig_len < max:
                    for _ in range(max-orig_len):
                        run.append(np.nan)

            lengths = [len(d[k][j]) for j in range(len(d[k]))]
            d[k] = np.array(d[k])
            if args.simulate and args.simulate == d["name"]:
                brain = d["brain"]["network"]
                with open("tmp.nndf", "w") as f:
                    f.write(brain)
                    f.close()
    if args.simulate:
        print(f"Simulating {d['name']}")
        if args.footprint_title is None:
            args.footprint_title = d["name"]
        os.system("python simulate.py --brain tmp.nndf --body quadruped.urdf --best")
        os.system(f"python footprint_diagram.py -r tmp -t {args.footprint_title}")
                
    num_runs = np.min([len(c["fitness_results"]) for c in data])
    print("\nNumber of runs: ", num_runs)

    bootst = args.do_bootstrap

    plot_mean_and_bootstrapped_ci_over_time([np.array(c["diversity_results"]) for c in data], [np.array(
        c["diversity_results"]) for c in data], [c["name"] for c in data], "Generation", "Average diversity", plot_bootstrap=bootst, title=f"{experiment_name} - Diversity")
    plot_mean_and_bootstrapped_ci_over_time([np.array(c["species_results"]) for c in data], [np.array(
        c["species_results"]) for c in data], [c["name"] for c in data], "Generation", "N Species", plot_bootstrap=bootst, title=f"{experiment_name} - Species")
    plot_mean_and_bootstrapped_ci_over_time([np.array(c["threshold_results"]) for c in data], [np.array(c["threshold_results"]) for c in data], [
                                            np.array(c["name"]) for c in data], "Generation", "Species Threshold", plot_bootstrap=bootst, title=f"{experiment_name} - Species Threshold")
    plot_mean_and_bootstrapped_ci_over_time([np.array(c["nodes_results"]) for c in data], [np.array(c["nodes_results"]) for c in data], [
                                            np.array(c["name"]) for c in data], "Generation", "Number of Nodes", plot_bootstrap=bootst, title=f"{experiment_name} - Number of Nodes")
    plot_mean_and_bootstrapped_ci_over_time([np.array(c["connections_results"]) for c in data], [np.array(
        c["connections_results"]) for c in data], [c["name"] for c in data], "Generation", "Number of Connections", plot_bootstrap=bootst, title=f"{experiment_name} - Number of Connections")

    

    plot_mean_and_bootstrapped_ci_over_time([np.array(c["fitness_results"]) for c in data], [np.array(
        c["fitness_results"]) for c in data], [c["name"] for c in data], "Generation", f"Best fitness (average of {num_runs} runs)", plot_bootstrap=bootst, title=f"{experiment_name} - Fitness")
    
    plt.legend()
    
    fig, ax = plt.subplots() # generate figure and axes
    plt.title("Max Fitness")
    plt.bar(np.arange(len(data)), [np.max(c["fitness_results"]) for c in data], align="center")
    plt.xticks(np.arange(len(data)), [c["name"] for c in data])
    fig, ax = plt.subplots() # generate figure and axes
    
    plt.title("Gens to converge")
    plt.bar(np.arange(len(data)), [np.mean(c["gens_to_converge"]) for c in data], align="center", yerr=[np.std(c["gens_to_converge"]) for c in data])
    plt.xticks(np.arange(len(data)), [c["name"] for c in data])

    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run search on the robot controller.')
    parser.add_argument('-b', '--do_bootstrap', action='store_true',
                        help='Show bootstrap CI on experiment plots.')
    parser.add_argument('-f', '--experiment_file',
                        action='store', help='Experiment results file.', required=True)
    parser.add_argument('-g', '--gens', action='store',
                        help='Show only experimental runs with this number of generations.')
    parser.add_argument('-s', '--simulate', action='store',
                        help='Simulate the best robot from the given condition')
    parser.add_argument('-t', '--footprint_title',
                        action='store', help='Footprint graph title. Defaults to robot name.')
    args = parser.parse_args()
    main(args)
