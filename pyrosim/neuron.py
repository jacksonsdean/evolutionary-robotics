import math
from numpy import sin

import pybullet
from activations import string_to_fn, tanh

import pyrosim.pyrosim as pyrosim

import pyrosim.constants as c

import constants as const
class NEURON: 

    def __init__(self,line):

        self.Determine_Name(line)

        self.Determine_Type(line)

        self.Search_For_Link_Name(line)

        self.Search_For_Joint_Name(line)
        
        self.Search_For_BodyID(line)
        
        self.Search_For_Activation(line)

        self.Set_Value(0.0)

    def Add_To_Value( self, value ):

        self.Set_Value( self.Get_Value() + value )

    def Get_Joint_Name(self):

        return self.jointName

    def Get_BodyID(self):

        return self.bodyID

    def Get_Link_Name(self):

        return self.linkName

    def Get_Name(self):

        return self.name

    def Get_Value(self):

        return self.value

    def Is_Sensor_Neuron(self):

        return self.type in [c.TOUCH_SENSOR_NEURON, c.ROTATIONAL_SENSOR_NEURON, c.LINK_VELOCITY_SENSOR_NEURON, c.BASE_VELOCITY_SENSOR_NEURON]
    
    def Is_CPG_Neuron(self):

        return self.type == c.CPG_NEURON

    def Is_Hidden_Neuron(self):

        return self.type == c.HIDDEN_NEURON

    def Is_Motor_Neuron(self):

        return self.type == c.MOTOR_NEURON

    def Print(self):

        # self.Print_Name()

        # self.Print_Type()

        self.Print_Value()

        # print("")

    def Set_Value(self,value):

        self.value = value

    def Update_Sensor_Neuron(self):
        if self.type == c.TOUCH_SENSOR_NEURON:
            val = pyrosim.Get_Touch_Sensor_Value_For_Link(self.Get_Link_Name())
        elif self.type == c.ROTATIONAL_SENSOR_NEURON:
            if self.Get_BodyID() == 0:
                raise Exception("Proprioceptive sensor neuron has no bodyID")
            val = pyrosim.Get_Rotational_Sensor_Value_For_Joint(self.Get_Joint_Name(), self.Get_BodyID())
        elif self.type == c.LINK_VELOCITY_SENSOR_NEURON:
            if self.Get_BodyID() == 0:
                raise Exception("Proprioceptive sensor neuron has no bodyID")
            val = pyrosim.Get_Velocity_Sensor_Value_For_Link(self.Get_Link_Name(), self.Get_BodyID())
        elif self.type == c.BASE_VELOCITY_SENSOR_NEURON:
            if self.Get_BodyID() == 0:
                raise Exception("Proprioceptive sensor neuron has no bodyID")
            val = pyrosim.Get_Base_Velocity_Sensor_Value(self.Get_BodyID())
            
        self.Set_Value(val)
        
    def Update_CPG_Neuron(self, step):
        self.Set_Value(sin(step))

    def Update_Hidden_Or_Motor_Neuron(self, neurons, synapses):
        for pre_post_neurons, synapse in synapses.items():
            if self.Get_Name() == neurons[pre_post_neurons[1]].Get_Name():
                weight = synapse.Get_Weight()
                pre_synaptic_value = neurons[pre_post_neurons[0]].Get_Value()
                self.Allow_Presynaptic_Neuron_To_Influence_Me(weight, pre_synaptic_value)
        self.Threshold()

    def Allow_Presynaptic_Neuron_To_Influence_Me(self, weight, pre_synaptic_value):
        result = weight*pre_synaptic_value
        self.Add_To_Value(result)
# -------------------------- Private methods -------------------------

    def Determine_Name(self,line):

        if "name" in line:

            splitLine = line.split('"')

            self.name = splitLine[1]

    def Determine_Type(self,line):

        if "touch_sensor" in line:

            self.type = c.TOUCH_SENSOR_NEURON

        elif "rotation_sensor" in line:

            self.type = c.ROTATIONAL_SENSOR_NEURON

        elif "link_velocity_sensor" in line:

            self.type = c.LINK_VELOCITY_SENSOR_NEURON
            
        elif "base_velocity_sensor" in line:

            self.type = c.BASE_VELOCITY_SENSOR_NEURON
            
        elif "cpg" in line:

            self.type = c.CPG_NEURON

        elif "motor" in line:

            self.type = c.MOTOR_NEURON

        else:

            self.type = c.HIDDEN_NEURON

    def Print_Name(self):

       print(self.name)

    def Print_Type(self):

       print(self.type)

    def Print_Value(self):

       print(self.value , " " , end="" )

    def Search_For_Joint_Name(self,line):

        if "jointName" in line:

            splitLine = line.split('"')

            self.jointName = splitLine[5]

    def Search_For_Link_Name(self,line):

        if "linkName" in line:

            splitLine = line.split('"')

            self.linkName = splitLine[5]

    def Search_For_BodyID(self,line):

        if "bodyID" in line:

            splitLine = line.split('"')

            self.bodyID = int(splitLine[splitLine.index( ' bodyID=')+1])
            
    def Search_For_Activation(self,line):

        if "activation" in line:

            splitLine = line.split('"')

            self.activation = splitLine[splitLine.index(' activation=')+1]
            self.activation = string_to_fn(self.activation)
        else:
            self.activation = tanh

    def Threshold(self):

        self.value =self.activation(self.value)
