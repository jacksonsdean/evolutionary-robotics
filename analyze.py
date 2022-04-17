import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from util import plot_mean_and_bootstrapped_ci_over_time
# %%
# backLegSensorValues =  np.load("data/backLegSensorValues.npy")
# frontLegSensorValues =  np.load("data/frontLegSensorValues.npy")
# plt.plot(backLegSensorValues , label="Back Leg", linewidth=3.5)
# plt.plot(frontLegSensorValues, label="Front Leg", linestyle="--")


# %%
def main(args):
    filename = "experiments/weight_mutation_rate_low_mid_results.json"
    if args.experiment_file:
        filename = args.experiment_file
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


    for i, d in enumerate(data):
        for k in keys:
            lengths = [len(d[k][j]) for j in range(len(d[k]))]
            max = np.max(lengths)
            max_index = lengths.index(max)
            for j, run in enumerate(d[k]):
                orig_len = len(run)
                if orig_len < max:
                    for x in range(max-orig_len):
                        run.append(d[k][max_index][orig_len+x])
                                    
            lengths = [len(d[k][j]) for j in range(len(d[k]))]
            d[k] = np.array(d[k])
            
    plot_mean_and_bootstrapped_ci_over_time([np.array(c["fitness_results"]) for c in data], [np.array(
        c["fitness_results"]) for c in data], [c["name"] for c in data], "Generation", "Best fitness", plot_bootstrap=bootst)
    plot_mean_and_bootstrapped_ci_over_time([np.array(c["diversity_results"]) for c in data], [np.array(
        c["diversity_results"]) for c in data], [c["name"] for c in data], "Generation", "Average diversity", plot_bootstrap=bootst)

    bootst = args.do_bootstrap
    # plot fitness
    # plot species
    plot_mean_and_bootstrapped_ci_over_time([np.array(c["species_results"]) for c in data], [np.array(
        c["species_results"]) for c in data], [c["name"] for c in data], "Generation", "N Species", plot_bootstrap=bootst)
    plot_mean_and_bootstrapped_ci_over_time([np.array(c["threshold_results"]) for c in data], [np.array(c["threshold_results"]) for c in data], [
                                            np.array(c["name"]) for c in data], "Generation", "Species Threshold", plot_bootstrap=bootst)

    plot_mean_and_bootstrapped_ci_over_time([np.array(c["nodes_results"]) for c in data], [np.array(c["nodes_results"]) for c in data], [
                                            np.array(c["name"]) for c in data], "Generation", "Number of Nodes", plot_bootstrap=bootst)
    plot_mean_and_bootstrapped_ci_over_time([np.array(c["connections_results"]) for c in data], [np.array(
        c["connections_results"]) for c in data], [c["name"] for c in data], "Generation", "Number of Connections", plot_bootstrap=bootst)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run search on the robot controller.')
    parser.add_argument('-b', '--do_bootstrap', action='store_true',
                        help='Show bootstrap CI on experiment plots.')
    parser.add_argument('-f', '--experiment_file',
                        action='store', help='Experiment results file.')
    args = parser.parse_args()
    main(args)
