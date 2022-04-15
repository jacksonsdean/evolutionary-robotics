import numpy as np
import matplotlib.pyplot as plt
import json
from util import plot_mean_and_bootstrapped_ci_over_time
#%%
# backLegSensorValues =  np.load("data/backLegSensorValues.npy")
# frontLegSensorValues =  np.load("data/frontLegSensorValues.npy")
# plt.plot(backLegSensorValues , label="Back Leg", linewidth=3.5)
# plt.plot(frontLegSensorValues, label="Front Leg", linestyle="--")

#%%
filename = "experiments/weight_mutation_rate_low_mid_results.json"
with open(filename) as f:
    data = json.load(f)


#%%
bootst = True
plot_mean_and_bootstrapped_ci_over_time([np.array(c["fitness_results"]) for c in data], [np.array(c["fitness_results"]) for c in data], [c["name"] for c in data], "Generation", "Best fitness", plot_bootstrap=bootst)
plot_mean_and_bootstrapped_ci_over_time([np.array(c["diversity_results"]) for c in data], [np.array(c["diversity_results"]) for c in data], [c["name"] for c in data], "Generation", "Average diversity", plot_bootstrap=bootst)

plt.legend()
plt.show()