import pybullet as p
import pybullet_data
import time
import pyrosim.pyrosim as pyrosim
import numpy as np

simulation_fps = 600
simulation_length = 1000

# create physics engine client
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # used by loadURDF

# set up gravity
p.setGravity(0,0,-9.8)

# load floor
planeId = p.loadURDF("plane.urdf")

# load robot
robotId = p.loadURDF("body.urdf")

# load world sdf file
p.loadSDF("world.sdf")

# prepare sensors for robot with id robotId
pyrosim.Prepare_To_Simulate(robotId)
backLegSensorValues = np.zeros(simulation_length)

step = 0
while True:
    try:
        p.stepSimulation()
        # sensors:
        backLegTouch = pyrosim.Get_Touch_Sensor_Value_For_Link("BackLeg")
        backLegSensorValues[step] = pyrosim.Get_Touch_Sensor
        # print("BackLeg:", backLegTouch)
        
        step+=1
        if step >= simulation_length:
            break
        time.sleep(1./simulation_fps) # sleep
        
    except KeyboardInterrupt:
        break
    except p.error as e:
        time.sleep(.01)
        print("\n","error in pybullet:", e)
        break
if p.isConnected():
    p.disconnect()

print(backLegSensorValues)