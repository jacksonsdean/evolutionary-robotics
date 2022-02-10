import random
import pybullet as p
import pybullet_data
import time
import pyrosim.pyrosim as pyrosim
import numpy as np

simulation_fps = 240
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
frontLegSensorValues = np.zeros(simulation_length)


# prepare motors
frontMotorAmplitude = np.pi/4.
frontMotorFreq = 400.
frontMotorPhaseOffset = np.pi*80.0/180.0

backMotorAmplitude = np.pi/4.
backMotorFreq = 10.0
backMotorPhaseOffset = 0.0

frontMotorTargetAngles = np.sin(np.linspace(-frontMotorFreq*np.pi + frontMotorPhaseOffset, frontMotorFreq*np.pi + frontMotorPhaseOffset, simulation_length)) * frontMotorAmplitude
backMotorTargetAngles = np.sin(np.linspace(-backMotorFreq*np.pi + backMotorPhaseOffset, backMotorFreq*np.pi + backMotorPhaseOffset, simulation_length)) * backMotorAmplitude
np.save("data/frontMotorTargetAngles.npy", frontMotorTargetAngles)
np.save("data/backMotorTargetAngles.npy", backMotorTargetAngles)

step = 0
while True:
    try:
        p.stepSimulation()
        # sensors:
        backLegSensorValues[step] = pyrosim.Get_Touch_Sensor_Value_For_Link("BackLeg")
        frontLegSensorValues[step] = pyrosim.Get_Touch_Sensor_Value_For_Link("FrontLeg")

        # motors
        pyrosim.Set_Motor_For_Joint(
            bodyIndex = robotId,
            jointName = "Torso_FrontLeg",
            controlMode = p.POSITION_CONTROL,
            targetPosition = frontMotorTargetAngles[step],
            maxForce = 100)
        pyrosim.Set_Motor_For_Joint(
            bodyIndex = robotId,
            jointName = "Torso_BackLeg",
            controlMode = p.POSITION_CONTROL,
            targetPosition = backMotorTargetAngles[step],
            maxForce = 100)
        
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
    
np.save("data/backLegSensorValues.npy", backLegSensorValues)
np.save("data/frontLegSensorValues.npy", frontLegSensorValues)