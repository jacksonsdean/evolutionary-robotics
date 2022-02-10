import numpy as np
import matplotlib.pyplot as plt

backLegSensorValues =  np.load("data/backLegSensorValues.npy")
frontLegSensorValues =  np.load("data/frontLegSensorValues.npy")
# plt.plot(backLegSensorValues , label="Back Leg", linewidth=3.5)
# plt.plot(frontLegSensorValues, label="Front Leg", linestyle="--")

frontMotorTargetAngles = np.load("data/frontMotorTargetAngles.npy")
backMotorTargetAngles = np.load("data/backMotorTargetAngles.npy")
plt.plot(frontMotorTargetAngles, label="Front Leg", linewidth=3.5)
plt.plot(backMotorTargetAngles, label="Back Leg", linestyle="--")

plt.legend()
plt.show()