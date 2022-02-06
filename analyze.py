import numpy as np
import matplotlib.pyplot as plt

# backLegSensorValues =  np.load("data/backLegSensorValues.npy")
# frontLegSensorValues =  np.load("data/frontLegSensorValues.npy")
# plt.plot(backLegSensorValues , label="Back Leg", linewidth=3.5)
# plt.plot(frontLegSensorValues, label="Front Leg", linestyle="--")

plt.plot(np.load("data/targetAngles.npy"), label="Target Angle", linewidth=3.5)


plt.legend()
plt.show()