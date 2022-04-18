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
    def __init__(self, headless_mode=False, solution_id=0, save_best=False, brain_path=None, body_path=None):
        self.solution_id = solution_id
        self.headless_mode = headless_mode
        # create physics engine client
        self.physicsClient = p.connect(p.DIRECT if self.headless_mode else p.GUI, )
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # used by loadURDF

        # set up gravity
        p.setGravity(*c.gravity)
  
        if Simulation.world is None:
            Simulation.world = World()
        self.robot = Robot(self.solution_id, brain_path, body_path, save_best)
        
        self.camera_lock = True
        
        if save_best:
            print("Saving best solution...")
            if platform.system() =="Windows":
                os.system(f"copy brain{self.solution_id}.nndf" + " best_brain.nndf")
                os.system(f"copy body{self.solution_id}.urdf" + " best_body.urdf")
            else:
                os.system(f"cp brain{self.solution_id}.nndf" + " best_brain.nndf")
                os.system(f"cp body{self.solution_id}.urdf" + " best_body.urdf")
            time.sleep(0.5)
            
        if platform.system() =="Windows":
            os.system(f"del brain{self.solution_id}.nndf")
            os.system(f"del body{self.solution_id}.urdf")
        else:
            os.system(f"rm brain{self.solution_id}.nndf")
            os.system(f"rm body{self.solution_id}.urdf")

        
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
                
                # check if c button pressed
                if not self.headless_mode:
                    events = p.getKeyboardEvents()
                    if 99 in events.keys() and events[99] == p.KEY_WAS_RELEASED:
                        self.camera_lock = not self.camera_lock
                        
                yaw = p.getDebugVisualizerCamera()[8]
                pitch = p.getDebugVisualizerCamera()[9]
                dist = p.getDebugVisualizerCamera()[10]
                
                if self.camera_lock:
                    p.resetDebugVisualizerCamera( cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=p.getBasePositionAndOrientation(self.robot.robotId)[0])
                
                step += 1

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
