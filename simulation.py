import random
import pybullet as p
import constants as c
import pybullet_data
import time
import pyrosim.pyrosim as pyrosim
import numpy as np

from robot import Robot
import os
import platform

from world import World
class Simulation():
    world = None
    def __init__(self, headless_mode=False, solution_id=0):
        self.solution_id = solution_id
        self.headless_mode = headless_mode
        # create physics engine client
        self.physicsClient = p.connect(p.DIRECT if self.headless_mode else p.GUI, )
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # used by loadURDF

        # set up gravity
        p.setGravity(*c.gravity)
  
        if Simulation.world is None:
            Simulation.world = World()
            
        self.robot = Robot(self.solution_id)
        if platform.system() =="Windows":
            os.system(f"del brain{self.solution_id}.nndf")
        else:
            os.system(f"rm brain{self.solution_id}.nndf")

        
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
                if c.simulation_fps > 0 and not self.headless_mode:
                    time.sleep(1./c.simulation_fps) # sleep
                
            except KeyboardInterrupt:
                break
            except p.error as e:
                print("\n","error in pybullet:", e)
                break

    def get_fitness(self):
        self.robot.get_fitness()
