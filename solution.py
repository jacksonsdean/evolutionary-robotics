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
    
    def start_simulation(self, headless, show_debug_output=False, save_as_best=False):
        self.generate_body()
        self.generate_brain()
        if platform.system() == "Windows":
            if show_debug_output:
                os.system(f"start /B python simulate.py {'DIRECT' if headless else 'GUI'} --id {self.id} {'--best' if save_as_best else ''}")
            else:
                os.system(f"start /B python simulate.py {'DIRECT' if headless else 'GUI'} --id {self.id} {'--best' if save_as_best else ''} > nul 2> nul")
                
        else:   
            if show_debug_output:
                os.system(f"python simulate.py {'DIRECT' if headless else 'GUI'} --id {self.id} {'--best' if save_as_best else ''}" + " &")
            else:
                os.system(f"python simulate.py {'DIRECT' if headless else 'GUI'} --id {self.id} {'--best' if save_as_best else ''} 2&>1" + " &")
            
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
        pyrosim.Send_Cube(name="FrontLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.])
        pyrosim.Send_Joint( name = "FrontLeg_FrontLowerLeg" , parent= "FrontLeg" , child = "FrontLowerLeg" , type = "revolute", position = [0,1,0], jointAxis = "1 0 0")
        pyrosim.Send_Cube(name="BackLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.])
        pyrosim.Send_Joint( name = "BackLeg_BackLowerLeg" , parent= "BackLeg" , child = "BackLowerLeg" , type = "revolute", position = [0,-1,0], jointAxis = "1 0 0")
        pyrosim.Send_Cube(name="LeftLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.])
        pyrosim.Send_Joint( name = "LeftLeg_LeftLowerLeg" , parent= "LeftLeg" , child = "LeftLowerLeg" , type = "revolute", position = [-1,0,0], jointAxis = "0 1 0")
        pyrosim.Send_Cube(name="RightLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.])
        pyrosim.Send_Joint( name = "RightLeg_RightLowerLeg" , parent= "RightLeg" , child = "RightLowerLeg" , type = "revolute", position = [1,0,0], jointAxis = "0 1 0")
        pyrosim.End()

    def generate_brain(self):
        pyrosim.Start_NeuralNetwork(f"brain{self.id}.nndf")
        
        # Neurons:
        # -Input
        n = 0
        # pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "Torso"); n+=1
        # pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "BackLeg"); n+=1
        # pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "FrontLeg"); n+=1
        # pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "LeftLeg"); n+=1
        # pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "RightLeg"); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , linkName = "FrontLowerLeg"); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , linkName = "BackLowerLeg"); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , linkName = "LeftLowerLeg"); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , linkName = "RightLowerLeg"); n+=1

        # -Hidden
        ...


        # -Output
        pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_BackLeg"); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_FrontLeg"); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_LeftLeg"); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_RightLeg"); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "FrontLeg_FrontLowerLeg"); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "BackLeg_BackLowerLeg"); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "LeftLeg_LeftLowerLeg"); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "RightLeg_RightLowerLeg"); n+=1

        # Synapses:
        # fully connected:
        for row in range(c.num_sensor_neurons):
            for col in range(c.num_motor_neurons):
                pyrosim.Send_Synapse(sourceNeuronName = row, targetNeuronName = col+c.num_sensor_neurons, weight = self.weights[row][col])

        pyrosim.End()
        
        while not os.path.exists(f"brain{self.id}.nndf"):
            time.sleep(0.01)

