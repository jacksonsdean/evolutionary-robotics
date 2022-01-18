import pybullet as p
import time

simulation_fps = 60

# create physics engine client
physicsClient = p.connect(p.GUI)
for step in range(1000):
    p.stepSimulation()
    time.sleep(1./simulation_fps)
p.disconnect()