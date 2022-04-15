import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from util import plot_mean_and_bootstrapped_ci_over_time
#%%
# backLegSensorValues =  np.load("data/backLegSensorValues.npy")
# frontLegSensorValues =  np.load("data/frontLegSensorValues.npy")
# plt.plot(backLegSensorValues , label="Back Leg", linewidth=3.5)
# plt.plot(frontLegSensorValues, label="Front Leg", linestyle="--")
args = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run search on the robot controller.')
    parser.add_argument('-b','--do_bootstrap', action='store_true', help='Show bootstrap CI on experiment plots.')
    parser.add_argument('-f','--experiment_file', action='store', help='Experiment results file.')
    args = parser.parse_args()


#%%
filename = "experiments/weight_mutation_rate_low_mid_results.json"
if args.experiment_file:
    filename = args.experiment_file
with open(filename) as f:
    data = json.load(f)


#%%
bootst = False
if args:
    bootst = args.do_bootstrap
plot_mean_and_bootstrapped_ci_over_time([np.array(c["fitness_results"]) for c in data], [np.array(c["fitness_results"]) for c in data], [c["name"] for c in data], "Generation", "Best fitness", plot_bootstrap=bootst)
plot_mean_and_bootstrapped_ci_over_time([np.array(c["diversity_results"]) for c in data], [np.array(c["diversity_results"]) for c in data], [c["name"] for c in data], "Generation", "Average diversity", plot_bootstrap=bootst)

plt.legend()
plt.show()