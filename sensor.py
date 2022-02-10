import numpy as np
import constants as c
import pyrosim.pyrosim as pyrosim
class Sensor():
    def __init__(self, linkName):
        self.linkName = linkName
        self.values = np.zeros(c.simulation_length)

    def GetValue(self, step):
        self.values[step] = pyrosim.Get_Touch_Sensor_Value_For_Link(self.linkName)
        return self.values[step]
    
    def SaveValues(self):
        np.save(f"data/{self.linkName}SensorValues.npy", self.values)