import random
import pybullet as p
import constants as c
import pybullet_data
import time
import pyrosim.pyrosim as pyrosim
import numpy as np

# create physics engine client
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # used by loadURDF

# set up gravity
p.setGravity(*c.gravity)

# load floor
planeId = p.loadURDF("plane.urdf")

# load robot
robotId = p.loadURDF("body.urdf")

# load world sdf file
p.loadSDF("world.sdf")

# prepare sensors for robot with id robotId
pyrosim.Prepare_To_Simulate(robotId)
backLegSensorValues = np.zeros(c.simulation_length)
frontLegSensorValues = np.zeros(c.simulation_length)


# prepare motors
frontMotorTargetAngles = np.sin(np.linspace(-c.frontMotorFreq*np.pi + c.frontMotorPhaseOffset, c.frontMotorFreq*np.pi + c.frontMotorPhaseOffset, c.simulation_length)) * c.frontMotorAmplitude
backMotorTargetAngles = np.sin(np.linspace(-c.backMotorFreq*np.pi + c.backMotorPhaseOffset, c.backMotorFreq*np.pi + c.backMotorPhaseOffset, c.simulation_length)) * c.backMotorAmplitude
np.save("data/frontMotorTargetAngles.npy", frontMotorTargetAngles)
np.save("data/backMotorTargetAngles.npy", backMotorTargetAngles)

step = 0
while True:
    try:
        p.stepSimulation() # step
        
        # sensors:
        backLegSensorValues[step] = pyrosim.Get_Touch_Sensor_Value_For_Link("BackLeg")
        frontLegSensorValues[step] = pyrosim.Get_Touch_Sensor_Value_For_Link("FrontLeg")

        # motors
        pyrosim.Set_Motor_For_Joint(
            bodyIndex = robotId,
            jointName = "Torso_FrontLeg",
            controlMode = p.POSITION_CONTROL,
            targetPosition = frontMotorTargetAngles[step],
            maxForce = c.front_max_force)
        pyrosim.Set_Motor_For_Joint(
            bodyIndex = robotId,
            jointName = "Torso_BackLeg",
            controlMode = p.POSITION_CONTROL,
            targetPosition = backMotorTargetAngles[step],
            maxForce = c.back_max_force)
        
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
if p.isConnected():
    p.disconnect()
    
np.save("data/backLegSensorValues.npy", backLegSensorValues)
np.save("data/frontLegSensorValues.npy", frontLegSensorValues)