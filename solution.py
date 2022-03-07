import time
import numpy as np
import pyrosim.pyrosim as pyrosim
import platform
import os

class Solution():
    def __init__(self, id):
        self.weights = np.random.rand(3, 2)
        self.weights = self.weights * 2. - 1.
        self.set_id(id)
    
    def start_simulation(self, headless):
        self.generate_brain()
        if platform.system() == "Windows":
            # os.system(f"conda activate evo-robots & start /B python simulate.py {'DIRECT' if headless else 'GUI'}")
            os.system(f"start /B python simulate.py {'DIRECT' if headless else 'GUI'} --id {self.id}")
            time.sleep(1)
        else:   
            os.system(f"python simulate.py {'DIRECT' if headless else 'GUI'} --id {self.id}" + " &")
            
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

    def evaluate(self, headless=True):
        # self.start_simulation(headless)
        # self.wait_for_simulation()
        ...

    def set_id(self, id):
        self.id = id

    def mutate(self):
        mutate_row = np.random.randint(0, 3)
        mutate_col = np.random.randint(0, 2)
        self.weights[mutate_row, mutate_col] = np.random.rand() * 2. - 1.

    def create_world(self):
        pyrosim.Start_SDF("world.sdf")
        pyrosim.Send_Cube(name="Box", pos=[x-2., y+2., z], size=[length, width, height])
        pyrosim.End()
        
    def generate_body(self):
        pyrosim.Start_URDF(f"body{self.id}.urdf")
        pyrosim.Send_Cube(name="Torso", pos=[1.5, 0, 1.5], size=[length, width, height])
        pyrosim.Send_Joint( name = "Torso_BackLeg" , parent= "Torso" , child = "BackLeg" , type = "revolute", position = [1,0,1])
        pyrosim.Send_Cube(name="BackLeg", pos=[-.5, 0, -.5], size=[length, width, height])
        pyrosim.Send_Joint( name = "Torso_FrontLeg" , parent= "Torso" , child = "FrontLeg" , type = "revolute", position = [2,0,1])
        pyrosim.Send_Cube(name="FrontLeg", pos=[.5, 0, -.5], size=[length, width, height])
        pyrosim.End()

    def generate_brain(self):
        pyrosim.Start_NeuralNetwork(f"brain{self.id}.nndf")
        
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
        for row in [0, 1, 2]:
            for col in [0, 1]:
                pyrosim.Send_Synapse(sourceNeuronName = row, targetNeuronName = col+3, weight = self.weights[row][col])

        pyrosim.End()
        
        while not os.path.exists(f"brain{self.id}.nndf"):
            time.sleep(0.01)

