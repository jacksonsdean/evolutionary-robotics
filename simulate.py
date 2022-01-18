import pybullet as p
import time

simulation_fps = 60

# create physics engine client
physicsClient = p.connect(p.GUI)
while True:
    try:
        p.stepSimulation()
        time.sleep(1./simulation_fps)
    except KeyboardInterrupt:
        break
p.disconnect()