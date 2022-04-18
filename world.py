import pybullet as p
import pyrosim.pyrosim as pyrosim
import constants as c
class World():
    def __init__(self):
        # load floor
        self.planeId = p.loadURDF("plane.urdf")

        if c.use_obstacles:
            for x in range(25, -25,-1):
                p.loadURDF(f"world/cube_{x}.urdf")
                
            for y in range(25, -25,-1):
                p.loadURDF(f"world/cube_y{y}.urdf")
                
        else:
        # load world sdf file
            p.loadSDF("world.sdf")
        