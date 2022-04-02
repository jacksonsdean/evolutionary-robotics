import os
import pybullet as p
import pyrosim.pyrosim as pyrosim
import numpy as np
import constants as c
from sensor import Sensor
from motor import Motor
from pyrosim.neuralNetwork import NEURAL_NETWORK
import platform

class Robot():
    def __init__(self, solution_id):
        self.solution_id = solution_id
        # load robot
        self.robotId = p.loadURDF(f"bodies/body{solution_id}.urdf")

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
        self.nn = NEURAL_NETWORK(f"brains/brain{self.solution_id}.nndf")

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
                self.motors[jointName].SetValue(desiredAngle * c.motor_joint_range)
    
    def get_fitness(self):
        basePositionAndOrientation = p.getBasePositionAndOrientation(self.robotId)
        basePosition = basePositionAndOrientation[0]
        xPos, yPos, zPos = basePosition
        with open(f"tmp{self.solution_id}.txt", "w") as f:
            fitness = -1.0 * xPos 
            f.write(str(fitness))
        
        f.close()
        if platform.system() == "Windows":
            os.rename("tmp"+str(self.solution_id)+".txt" , "fitness"+str(self.solution_id)+".txt")
        else:
            os.system(f"mv tmp{self.solution_id}.txt fitness{self.solution_id}.txt")