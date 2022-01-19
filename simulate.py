import pybullet as p
import pybullet_data
import time

simulation_fps = 60

# create physics engine client
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # used by loadURDF

# set up gravity
p.setGravity(0,0,-9.8)

# add floor
planeId = p.loadURDF("plane.urdf")

# load world sdf file
p.loadSDF("boxes.sdf")

while True:
    try:
        p.stepSimulation()
        time.sleep(1./simulation_fps)
    except KeyboardInterrupt:
        break
    except p.error as e:
        time.sleep(.01)
        print("\n","error in pybullet:", e)
        break
if p.isConnected():
    p.disconnect()