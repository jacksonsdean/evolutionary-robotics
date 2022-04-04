import pybullet as p
import pyrosim.pyrosim as pyrosim
import random
class World():
    def __init__(self):
        # load floor
        self.planeId = p.loadURDF("plane.urdf")

        for x in range(-2, -25,-1):
            p.loadURDF(f"world/cube_{x}.urdf")
            
        
        # load world sdf file
        # p.loadSDF("world.sdf")
        