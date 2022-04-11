import random
import pyrosim.pyrosim as pyrosim

import constants as c

def Create_World():
    for x in range(25, -25,-1):
        pyrosim.Start_URDF(f"world/cube_{x}.urdf")
        y = random.random()
        z_size = random.random() * c.max_obstacle_height + .01
        pyrosim.Send_Cube(name=f"Cube{x}", pos=[x, y, z_size/2.], size=[.5, 600, z_size], static=True, color_name="dark_gray" if x % 2 == 0 else "gray", color_rgba= [0.7, 0.7, 0.7, 1.0] if x % 2 == 0 else [1, 1, 1, 1.0], mass=0.0)
        pyrosim.End()
    for y in range(25, -25,-1):
        pyrosim.Start_URDF(f"world/cube_y{y}.urdf")
        x = random.random()
        z_size = random.random() * c.max_obstacle_height + .01
        pyrosim.Send_Cube(name=f"Cube{x}", pos=[x, y, z_size/2.], size=[600, .5, z_size], static=True, color_name="dark_gray" if y % 2 == 0 else "gray", color_rgba= [0.7, 0.7, 0.7, 1.0] if y % 2 == 0 else [1, 1, 1, 1.0], mass=0.0)
        pyrosim.End()
        
    
def Generate_Body():
        pyrosim.Start_URDF(f"body.urdf")
        pyrosim.Send_Cube(name="Torso", pos=[0, 0, 1], size=[1, 1, 1], mass=c.torso_weight)
        pyrosim.Send_Joint( name = "Torso_BackLegRot" , parent= "Torso" , child = "BackLegRot" , type = "revolute", position = [0, -0.5, 1.0], jointAxis = "0 1 0")
        pyrosim.Send_Joint( name = "BackLegRot_BackLeg" , parent= "BackLegRot" , child = "BackLeg" , type = "revolute", position = [0, 0, 0], jointAxis = "1 0 0")
        pyrosim.Send_Cube(name="BackLegRot", pos=[0.0, -0.5, 0.0], size=[0,0,0], mass=0.0)
        pyrosim.Send_Cube(name="BackLeg", pos=[0.0, -0.5, 0.0], size=[.2, 1., .2], mass=1.0)
        pyrosim.Send_Joint( name = "Torso_FrontLegRot" , parent= "Torso" , child = "FrontLegRot" , type = "revolute", position = [0.0, 0.5, 1.0], jointAxis = "1 0 0")
        pyrosim.Send_Joint( name ="FrontLegRot_FrontLeg" , parent= "FrontLegRot" , child = "FrontLeg" , type = "revolute", position = [0.0, 0.0, 0.0], jointAxis = "0 1 0")
        pyrosim.Send_Cube(name="FrontLegRot", pos=[0.0, 0.5, 0], size=[0,0,0], mass=0.0)
        pyrosim.Send_Cube(name="FrontLeg", pos=[0.0, 0.5, 0], size=[.2, 1., .2], mass=1.0)
        pyrosim.Send_Cube(name="LeftLeg", pos=[-0.5, 0.0, 0.0], size=[1.0, 0.2, 0.2], mass=1.0)
        pyrosim.Send_Cube(name="LeftLegRot", pos=[-0.5, 0.0, 0.0], size=[0,0,0], mass=0.0)
        pyrosim.Send_Joint( name = "Torso_LeftLegRot" , parent= "Torso" , child = "LeftLegRot" , type = "revolute", position = [-0.5, 0, 1.], jointAxis = "1 0 0")
        pyrosim.Send_Joint( name = "LeftLegRot_LeftLeg" , parent= "LeftLegRot" , child = "LeftLeg" , type = "revolute", position = [0,0,0], jointAxis = "0 1 0")
        pyrosim.Send_Cube(name="RightLegRot", pos=[0.5, 0.0, 0.0], size=[0,0,0], mass=0.0)
        pyrosim.Send_Cube(name="RightLeg", pos=[0.5, 0.0, 0.0], size=[1.0, 0.2, 0.2], mass=1.0)
        pyrosim.Send_Joint( name = "Torso_RightLegRot" , parent= "Torso" , child = "RightLegRot" , type = "revolute", position = [0.5, 0, 1.], jointAxis = "1 0 0")
        pyrosim.Send_Joint( name = "RightLegRot_RightLeg" , parent= "RightLegRot" , child = "RightLeg" , type = "revolute", position = [0,0,0], jointAxis = "0 1 0")
        pyrosim.Send_Cube(name="FrontLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.], mass=1.0)
        pyrosim.Send_Joint( name = "FrontLeg_FrontLowerLeg" , parent= "FrontLeg" , child = "FrontLowerLeg" , type = "revolute", position = [0,1,0], jointAxis = "1 0 0")
        pyrosim.Send_Cube(name="BackLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.], mass=1.0)
        pyrosim.Send_Joint( name = "BackLeg_BackLowerLeg" , parent= "BackLeg" , child = "BackLowerLeg" , type = "revolute", position = [0,-1,0], jointAxis = "1 0 0")
        pyrosim.Send_Cube(name="LeftLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.], mass=1.0)
        pyrosim.Send_Joint( name = "LeftLeg_LeftLowerLeg" , parent= "LeftLeg" , child = "LeftLowerLeg" , type = "revolute", position = [-1,0,0], jointAxis = "0 1 0")
        pyrosim.Send_Cube(name="RightLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.], mass=1.0)
        pyrosim.Send_Joint( name = "RightLeg_RightLowerLeg" , parent= "RightLeg" , child = "RightLowerLeg" , type = "revolute", position = [1,0,0], jointAxis = "0 1 0")
        pyrosim.End()

def Generate_Brain():
    pyrosim.Start_NeuralNetwork(f"brain.nndf")
    
    # Neurons:
    # -Input
    n = 0
    pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "FrontLowerLeg"); n+=1
    pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "BackLowerLeg"); n+=1
    pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "LeftLowerLeg"); n+=1
    pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "RightLowerLeg"); n+=1
    
    pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "Torso_BackLegRot", bodyID=1); n+=1
    pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "Torso_FrontLegRot", bodyID=1); n+=1
    pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "Torso_LeftLegRot", bodyID=1); n+=1
    pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "Torso_RightLegRot", bodyID=1); n+=1

    # -Hidden
    ...
                
    # -Output
    pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_BackLegRot"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "BackLegRot_BackLeg"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_FrontLegRot"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "FrontLegRot_FrontLeg"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_LeftLegRot"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "LeftLegRot_LeftLeg"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_RightLegRot"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "RightLegRot_RightLeg"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "FrontLeg_FrontLowerLeg"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "BackLeg_BackLowerLeg"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "LeftLeg_LeftLowerLeg"); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "RightLeg_RightLowerLeg"); n+=1


    # Synapses:
    # fully connected:
    for i in [0, 1, 2]:
        for j in [3, 4]:
            pyrosim.Send_Synapse(sourceNeuronName = i, targetNeuronName = j, weight = 2.*random.random()-1.)

    pyrosim.End()

def Create_Robot():
    Generate_Body()
    Generate_Brain()


def Generate():
    print("Generating world and robot")
    Create_World()
    Create_Robot()

if __name__ == '__main__':
    Generate()