import pybullet as p
import pyrosim.pyrosim as pyrosim
import numpy as np
import constants as c
from sensor import Sensor
from motor import Motor
from pyrosim.neuralNetwork import NEURAL_NETWORK
class Robot():
    def __init__(self):
        # load robot
        self.robotId = p.loadURDF("body.urdf")

        pyrosim.Prepare_To_Simulate(self.robotId)
        self.PrepareSensors()
        self.PrepareBrain()
        self.PrepareMotors()

    def PrepareSensors(self):
        # prepare sensors for robot with id robotId
        self.sensors = {}
        for linkName in pyrosim.linkNamesToIndices:
            self.sensors[linkName] = Sensor(linkName)
    
    def PrepareBrain(self):
        self.nn = NEURAL_NETWORK("brain.nndf")

    def PrepareMotors(self):
        # prepare motors
        self.motors = {}
        for jointName in pyrosim.jointNamesToIndices:
            self.motors[jointName] = Motor(jointName, self.robotId)
        
    def Sense(self, step):
        # sensors:
        for sensor in self.sensors.values():
            sensor.GetValue(step)

    def Think(self, step):
        self.nn.Update(step)
    
    def Act(self, step):
        for neuronName in self.nn.Get_Neuron_Names():
            if self.nn.Is_Motor_Neuron(neuronName):
                jointName = self.nn.Get_Motor_Neurons_Joint(neuronName)
                desiredAngle = self.nn.Get_Value_Of(neuronName)
                self.motors[jointName].SetValue(desiredAngle)
    
    def get_fitness(self):
        stateOfFirstLink = p.getLinkState(self.robotId, 0)
        positionOfFirstLink = stateOfFirstLink[0]
        xPos = positionOfFirstLink[0]
        with open("fitness.txt", "w") as f:
            f.write(str(xPos))
        f.close()