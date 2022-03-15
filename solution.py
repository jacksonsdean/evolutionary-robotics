import time
import numpy as np
import pyrosim.pyrosim as pyrosim
import platform
import os
import constants as c

class Solution():
    def __init__(self, id):
        self.weights = np.random.rand(c.num_sensor_neurons, c.num_motor_neurons)
        self.weights = self.weights * 2. - 1.
        self.set_id(id)
    
    def start_simulation(self, headless):
        self.generate_body()
        self.generate_brain()
        if platform.system() == "Windows":
            # os.system(f"conda activate evo-robots & start /B python simulate.py {'DIRECT' if headless else 'GUI'}")
            os.system(f"start /B python simulate.py {'DIRECT' if headless else 'GUI'} --id {self.id} > nul 2> nul")
        else:   
            os.system(f"python simulate.py {'DIRECT' if headless else 'GUI'} --id {self.id} 2&>1" + " &")
            
    def wait_for_simulation(self):
        fit_file = f"fitness{self.id}.txt"

        while not os.path.exists(fit_file):
            time.sleep(0.01)

        with open(fit_file) as f:
            self.fitness = float(f.read())
            
        if platform.system() == "Windows":
            os.system(f"del fitness{self.id}.txt")
        else:
            os.system(f"rm fitness{self.id}.txt")
        
        f.close()

    def set_id(self, id):
        self.id = id

    def mutate(self):
        mutate_row = np.random.randint(0, 3)
        mutate_col = np.random.randint(0, 2)
        self.weights[mutate_row, mutate_col] = np.random.rand() * 2. - 1.

    def generate_body(self):
        pyrosim.Start_URDF(f"body{self.id}.urdf")
        pyrosim.Send_Cube(name="Torso", pos=[0, 0, 1], size=[1, 1, 1])
        pyrosim.Send_Joint( name = "Torso_BackLeg" , parent= "Torso" , child = "BackLeg" , type = "revolute", position = [0, -0.5, 1.0], jointAxis = "1 0 0")
        pyrosim.Send_Cube(name="BackLeg", pos=[0.0, -0.5, 0.0], size=[.2, 1., .2])
        pyrosim.Send_Joint( name = "Torso_FrontLeg" , parent= "Torso" , child = "FrontLeg" , type = "revolute", position = [0.0, 0.5, 1.0], jointAxis = "1 0 0")
        pyrosim.Send_Cube(name="FrontLeg", pos=[0.0, 0.5, 0], size=[.2, 1., .2])
        pyrosim.Send_Cube(name="LeftLeg", pos=[-0.5, 0.0, 0.0], size=[1.0, 0.2, 0.2])
        pyrosim.Send_Joint( name = "Torso_LeftLeg" , parent= "Torso" , child = "LeftLeg" , type = "revolute", position = [-0.5, 0, 1.], jointAxis = "0 1 0")
        pyrosim.Send_Cube(name="RightLeg", pos=[0.5, 0.0, 0.0], size=[1.0, 0.2, 0.2])
        pyrosim.Send_Joint( name = "Torso_RightLeg" , parent= "Torso" , child = "RightLeg" , type = "revolute", position = [0.5, 0, 1.], jointAxis = "0 1 0")
        pyrosim.End()

    def generate_brain(self):
        pyrosim.Start_NeuralNetwork(f"brain{self.id}.nndf")
        
        # Neurons:
        # -Input
        pyrosim.Send_Sensor_Neuron(name = 0 , linkName = "Torso")
        pyrosim.Send_Sensor_Neuron(name = 1 , linkName = "BackLeg")
        pyrosim.Send_Sensor_Neuron(name = 2 , linkName = "FrontLeg")
        pyrosim.Send_Sensor_Neuron(name = 3 , linkName = "LeftLeg")
        pyrosim.Send_Sensor_Neuron(name = 4 , linkName = "RightLeg")

        # -Hidden
        ...


        # -Output
        pyrosim.Send_Motor_Neuron( name = 3 , jointName = "Torso_BackLeg")
        pyrosim.Send_Motor_Neuron( name = 4 , jointName = "Torso_FrontLeg")
        pyrosim.Send_Motor_Neuron( name = 5 , jointName = "Torso_LeftLeg")
        pyrosim.Send_Motor_Neuron( name = 6 , jointName = "Torso_RightLeg")

        # Synapses:
        # fully connected:
        for row in range(c.num_sensor_neurons):
            for col in range(c.num_motor_neurons):
                pyrosim.Send_Synapse(sourceNeuronName = row, targetNeuronName = col+3, weight = self.weights[row][col])

        pyrosim.End()
        
        while not os.path.exists(f"brain{self.id}.nndf"):
            time.sleep(0.01)

