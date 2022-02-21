import random
import pybullet as p
import constants as c
import pybullet_data
import time
import pyrosim.pyrosim as pyrosim
import numpy as np

from robot import Robot
from world import World

class Simulation():
    def __init__(self):
        # create physics engine client
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # used by loadURDF

        # set up gravity
        p.setGravity(*c.gravity)
  
        self.world = World()
        self.robot = Robot()
        
    def __del__(self):
        if p.isConnected():
            p.disconnect()
         
        
    def run(self):
        step = 0
        while True:
            try:
                p.stepSimulation() # step
                self.robot.Sense(step) # update sensors
                self.robot.Think(step) # update neurons
                self.robot.Act(step) # update motors
                
                step+=1
                if step >= c.simulation_length:
                    # full simulation time has elapsed
                    break
                time.sleep(1./c.simulation_fps) # sleep
                
            except KeyboardInterrupt:
                break
            except p.error as e:
                print("\n","error in pybullet:", e)
                break
        
