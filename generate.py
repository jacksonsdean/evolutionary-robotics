import random
import pyrosim.pyrosim as pyrosim


x, y, z = 0, 0, 0.5
length, width, height = 1, 1, 1

def Create_World():
    for x in range(-2, -25,-1):
        pyrosim.Start_URDF(f"world/cube_{x}.urdf")
        y = random.random()
        z_size = random.random() * .25 + .01
        pyrosim.Send_Cube(name=f"Cube{x}", pos=[x, y, z_size/2.], size=[1, 600, z_size], static=True, color_name="dark_gray" if x % 2 == 0 else "gray", color_rgba= [0.7, 0.7, 0.7, 1.0] if x % 2 == 0 else [1, 1, 1, 1.0], mass=0.0)
        pyrosim.End()
        
    
def Generate_Body():
    pyrosim.Start_URDF("body.urdf")
    pyrosim.Send_Cube(name="Torso", pos=[1.5, 0, 1.5], size=[length, width, height])
    pyrosim.Send_Joint( name = "Torso_BackLeg" , parent= "Torso" , child = "BackLeg" , type = "revolute", position = [1,0,1])
    pyrosim.Send_Cube(name="BackLeg", pos=[-.5, 0, -.5], size=[length, width, height])
    pyrosim.Send_Joint( name = "Torso_FrontLeg" , parent= "Torso" , child = "FrontLeg" , type = "revolute", position = [2,0,1])
    pyrosim.Send_Cube(name="FrontLeg", pos=[.5, 0, -.5], size=[length, width, height])
    pyrosim.End()

def Generate_Brain():
    pyrosim.Start_NeuralNetwork("brain.nndf")
    
    # Neurons:
    # -Input
    pyrosim.Send_Sensor_Neuron(name = 0 , linkName = "Torso")
    pyrosim.Send_Sensor_Neuron(name = 1 , linkName = "BackLeg")
    pyrosim.Send_Sensor_Neuron(name = 2 , linkName = "FrontLeg")

    # -Hidden
    ...


    # -Output
    pyrosim.Send_Motor_Neuron( name = 3 , jointName = "Torso_BackLeg")
    pyrosim.Send_Motor_Neuron( name = 4 , jointName = "Torso_FrontLeg")

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