import pybullet as p
import time

simulation_fps = 60

# create physics engine client
physicsClient = p.connect(p.GUI)
p.loadSDF("box.sdf")
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