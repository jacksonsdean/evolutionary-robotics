import argparse
import os
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
    brain = None
    experiment_name = filename.split("/")[-1].split(".")[0]
    if "args" in data[0].keys():
        experiment_config = data[0]["args"]["experiment_file"]
        with open(experiment_config) as f:
            config = json.load(f)
            experiment_name = config["name"]

    for i, d in enumerate(data):
        for k in keys:
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
    print(f"Simulating {d['name']}")
    os.system("python simulate.py --brain tmp.nndf --body best_body.urdf")
                
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
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run search on the robot controller.')
    parser.add_argument('-b', '--do_bootstrap', action='store_true',
                        help='Show bootstrap CI on experiment plots.')
    parser.add_argument('-f', '--experiment_file',
                        action='store', help='Experiment results file.')
    parser.add_argument('-g', '--gens', action='store',
                        help='Show only experimental runs with this number of generations.')
    parser.add_argument('-s', '--simulate', action='store',
                        help='Simulate the best robot from the given condition')
    args = parser.parse_args()
    main(args)
