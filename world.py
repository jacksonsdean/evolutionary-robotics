import pybullet as p

class World():
    def __init__(self):
        # load floor
        self.planeId = p.loadURDF("plane.urdf")

        # load world sdf file
        p.loadSDF("world.sdf")
        