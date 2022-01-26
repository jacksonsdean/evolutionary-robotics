import numpy as np
import matplotlib.pyplot as plt

backLegSensorValues =  np.load("data/backLegSensorValues.npy")
plt.plot(backLegSensorValues)
plt.show()