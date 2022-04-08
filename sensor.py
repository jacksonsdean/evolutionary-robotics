import numpy as np
import constants as c
import pyrosim.pyrosim as pyrosim
import pybullet as p
class Sensor():
    def __init__(self, linkName):
        self.linkName = linkName
        self.values = np.zeros(c.simulation_length)

    def GetValue(self, step):
        self.values[step] = pyrosim.Get_Touch_Sensor_Value_For_Link(self.linkName)
        return self.values[step]
    
    def SaveValues(self):
        np.save(f"data/{self.linkName}SensorValues.npy", self.values)
        
class TorqueSensor(Sensor):
    def __init__(self, linkName, bodyID):
        self.linkName = linkName
        self.bodyID = bodyID
        self.values = np.zeros(c.simulation_length)
    
    def GetValue(self, step):
        self.values[step] = pyrosim.Get_Rotational_Sensor_Value_For_Joint(self.linkName, self.bodyID)
        return self.values[step]
    
class OrientationSensor(Sensor):
    def __init__(self, linkName, bodyID):
        self.linkName = linkName
        self.bodyID = bodyID
        self.values = np.zeros(c.simulation_length)
        self.axis = 0
    
    def GetValue(self, step):
        v = p.getLinkState(self.bodyID, pyrosim.linkNameToIndices[self.linkName])[1]
        self.values[step] = p.getEulerFromQuaternion(v)[self.axis]
        return self.values[step]
    