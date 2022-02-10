import numpy as np
import constants as c
import pyrosim.pyrosim as pyrosim
import pybullet as p

class Motor():
    def __init__(self, jointName, robotId):
        self.jointName = jointName
        self.robotId = robotId
        self.phaseOffset = c.frontMotorPhaseOffset
        self.amplitude = c.frontMotorAmplitude
        self.freq = c.frontMotorFreq
        
        self.motorValues = np.sin(np.linspace(-self.freq*np.pi + self.phaseOffset, self.freq*np.pi + self.phaseOffset, c.simulation_length)) * self.amplitude
    
    def SetValue(self, step):
        pyrosim.Set_Motor_For_Joint(
                    bodyIndex = self.robotId,
                    jointName = self.jointName,
                    controlMode = p.POSITION_CONTROL,
                    targetPosition = self.motorValues[step],
                    maxForce = c.front_max_force)
        
    def SaveValues(self):
       np.save(f"data/{self.jointName}MotorValues.npy", self.motorValues)
