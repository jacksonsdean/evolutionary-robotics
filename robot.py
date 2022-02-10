import pybullet as p
import pyrosim.pyrosim as pyrosim
import numpy as np
import constants as c
from sensor import Sensor
from motor import Motor
class Robot():
    def __init__(self):
        # load robot
        self.robotId = p.loadURDF("body.urdf")

        pyrosim.Prepare_To_Simulate(self.robotId)
        self.PrepareSensors()
        self.PrepareMotors()
        
    def PrepareSensors(self):
        # prepare sensors for robot with id robotId
        self.sensors = {}
        for linkName in pyrosim.linkNamesToIndices:
            self.sensors[linkName] = Sensor(linkName)

    def PrepareMotors(self):
        # prepare motors
        self.motors = {}
        for jointName in pyrosim.jointNamesToIndices:
            self.motors[jointName] = Motor(jointName, self.robotId)
        
    def Sense(self, step):
        # sensors:
        for sensor in self.sensors.values():
            sensor.GetValue(step)

    def Act(self, step):
        for motor in self.motors.values():
            motor.SetValue(step)