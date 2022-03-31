import random
import time
import numpy as np
import pyrosim.pyrosim as pyrosim
import platform
import os
import constants as c
from neat_genome import Genome, Config

class NeatSolution():
    def __init__(self, id):
        self.weights = np.random.rand(c.num_sensor_neurons, c.num_motor_neurons)
        self.weights = self.weights * 2. - 1.
        self.set_id(id)
        self.config = Config()
        self.config.num_inputs = c.num_sensor_neurons
        self.config.num_outputs = c.num_motor_neurons
        self.genome = Genome(self.config)
    
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
        if random.random() < self.config.prob_add_connection:
            self.genome.add_connection()
        if random.random() < self.config.prob_add_node:
            self.genome.add_node()
        if random.random() < self.config.prob_remove_node:
            self.genome.remove_node()
        if random.random() < self.config.prob_disable_connection:
            self.genome.disable_connection()
            
        self.genome.mutate_weights_with_prob()

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
        pyrosim.Send_Sensor_Neuron(name = n , linkName = "FrontLowerLeg"); n+=1
        pyrosim.Send_Sensor_Neuron(name = n , linkName = "BackLowerLeg"); n+=1
        pyrosim.Send_Sensor_Neuron(name = n , linkName = "LeftLowerLeg"); n+=1
        pyrosim.Send_Sensor_Neuron(name = n , linkName = "RightLowerLeg"); n+=1

        # -Hidden
        for neuron in self.genome.hidden_nodes():
            pyrosim.Send_Hidden_Neuron(name = neuron.id); 
            


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
        for synapse in self.genome.connection_genome:
                pyrosim.Send_Synapse(sourceNeuronName = synapse.fromNode.id, targetNeuronName = synapse.toNode.id, weight = synapse.weight)

        pyrosim.End()
        
        while not os.path.exists(f"brain{self.id}.nndf"):
            time.sleep(0.01)

