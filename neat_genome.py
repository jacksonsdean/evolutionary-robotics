import copy
from enum import IntEnum
import math
import uuid
import json
import numpy as np
import time
import numpy as np
import pyrosim.pyrosim as pyrosim
import platform
import os, os.path
import constants as c
from util import visualize_network

from util import choose_random_function, visualize_network

class NodeType(IntEnum):
    Input = 0
    Output = 1
    Hidden = 2

class Node:
    def __init__(self, fn, _type, _id, _layer=2) -> None:
        self.fn = fn
        self.uuid = uuid.uuid1()
        self.id = _id
        self.type = _type
        self.layer = _layer
        self.sum_inputs = np.zeros(1)
        self.outputs = np.zeros(1)
        self.sum_input = 0
        self.output = 0

    def empty():
        return Node(c.tanh, NodeType.Hidden, 0, 0)


class Connection:
    # connection            e.g.  2->5,  1->4
    # innovation_number            0      1
    # where innovation number is the same for all of same connection
    # i.e. 2->5 and 2->5 have same innovation number, regardless of Network
    # innovations = []
    current_innovation = 0

    def get_innovation():
        Connection.current_innovation += 1
        return Connection.current_innovation

    def __init__(self, fromNode, toNode, weight, enabled=True) -> None:
        self.fromNode = fromNode  # TODO change to node ids?
        self.toNode = toNode
        self.weight = weight
        self.innovation = Connection.get_innovation()
        self.enabled = enabled
        self.is_recurrent = toNode.layer < fromNode.layer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"([{self.fromNode.id}->{self.toNode.id}]I:{self.innovation} W:{self.weight:3f})"


def get_disjoint_connections(this_cxs, other_innovation):
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if (t_cx.innovation not in other_innovation and t_cx.innovation < other_innovation[-1])]


def get_excess_connections(this_cxs, other_innovation):
    if(len(this_cxs) == 0 or len(other_innovation) == 0):
        return []
    return [t_cx for t_cx in this_cxs if (t_cx.innovation not in other_innovation and t_cx.innovation > other_innovation[-1])]


def get_matching_connections(cxs_1, cxs_2):
    # returns connections in cxs_1 that share an innovation number with a connection in cxs_2
    # and     connections in cxs_2 that share an innovation number with a connection in cxs_1
    return sorted([c1 for c1 in cxs_1 if c1.innovation in [c2.innovation for c2 in cxs_2]], key=lambda x: x.innovation),\
        sorted([c2 for c2 in cxs_2 if c2.innovation in [
               c1.innovation for c1 in cxs_1]], key=lambda x: x.innovation)


class Genome:
    pixel_inputs = None
    current_id = 0
    
    def get_id():
        output = Genome.current_id
        Genome.current_id+=1
        return output
       
    def __init__(self) -> None:
        self.fitness = -math.inf
        self.novelty = -math.inf
        self.adjusted_fitness = -math.inf
        self.species_id = -1
        self.image = None
        self.node_genome = []  # inputs first, then outputs, then hidden
        self.connection_genome = []
        self.id = Genome.get_id()

        self.more_fit_parent = None  # for record-keeping
        self.n_hidden_nodes = c.hidden_nodes_at_start
        self.n_inputs = c.num_sensor_neurons
        self.n_outputs = c.num_motor_neurons
        total_node_count = c.num_sensor_neurons + \
            c.num_motor_neurons + c.hidden_nodes_at_start
        self.max_weight = c.max_weight
        self.weight_threshold = c.weight_threshold
        self.use_input_bias = c.use_input_bias
        self.allow_recurrent = c.allow_recurrent

        for i in range(c.num_sensor_neurons):
            self.node_genome.append(
                Node(choose_random_function(c), NodeType.Input, self.get_new_node_id(), 0))
        for i in range(c.num_sensor_neurons, c.num_sensor_neurons + c.num_motor_neurons):
            output_fn = choose_random_function(
                c) if c.output_activation is None else c.output_activation
            self.node_genome.append(
                Node(output_fn, NodeType.Output, self.get_new_node_id(), 2))
        for i in range(c.num_sensor_neurons + c.num_motor_neurons, total_node_count):
            self.node_genome.append(Node(choose_random_function(
                c), NodeType.Hidden, self.get_new_node_id(), 1))

        # initialize connection genome
        if self.n_hidden_nodes == 0:
            # connect all input nodes to all output nodes
            for input_node in self.input_nodes():
                for output_node in self.output_nodes():
                    new_cx = Connection(
                        input_node, output_node, self.random_weight())
                    self.connection_genome.append(new_cx)
                    if(np.random.rand() > c.init_connection_probability):
                        new_cx.enabled = False
        else:
           # connect all input nodes to all hidden nodes
            for input_node in self.input_nodes():
                for hidden_node in self.hidden_nodes():
                    new_cx = Connection(
                        input_node, hidden_node, self.random_weight())
                    self.connection_genome.append(new_cx)
                    if(np.random.rand() > c.init_connection_probability):
                        new_cx.enabled = False

           # connect all hidden nodes to all output nodes
            for hidden_node in self.hidden_nodes():
                for output_node in self.output_nodes():
                    if(np.random.rand() < c.init_connection_probability):
                        self.connection_genome.append(Connection(
                            hidden_node, output_node, self.random_weight()))

        for i in range(c.hidden_nodes_at_start-1):
            # TODO
            self.add_node()

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

        while not os.access(fit_file, os.F_OK):
            time.sleep(0.01)
        try:
            with open(fit_file) as f:
                self.fitness = float(f.read())
            f.close()
        except PermissionError as e:
            time.sleep(1)
            with open(fit_file) as f:
                self.fitness = float(f.read())
            f.close()

        time.sleep(0.1)
            
        if platform.system() == "Windows":
            os.system(f"del fitness{self.id}.txt")
        else:
            os.system(f"rm fitness{self.id}.txt")
        

    def set_id(self, id):
        self.id = id

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
        for neuron in self.hidden_nodes():
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
        for synapse in self.connection_genome:
                pyrosim.Send_Synapse(sourceNeuronName = synapse.fromNode.id, targetNeuronName = synapse.toNode.id, weight = synapse.weight)

        pyrosim.End()
        
        while not os.path.exists(f"brain{self.id}.nndf"):
            time.sleep(0.01)
            
        if False:
            num = len([n for n in os.listdir('tmp') if os.path.isfile(n)])
            os.system(f"copy brain{self.id}.nndf tmp\\{self.id}.nndf")
            visualize_network(self, sample=True, sample_point=[0.1, -0.1, .25, -.25], use_radial_distance=False, save_name=f"tmp/{self.id}_{num}.png", show_weights=False)



    def random_weight(self):
        return np.random.uniform(-self.max_weight, self.max_weight)

    def get_new_node_id(self):
        new_id = 0
        while len(self.node_genome) > 0 and new_id in [node.id for node in self.node_genome]:
            new_id += 1
        return new_id

    def update_with_fitness(self, fit, num_in_species):
        self.fitness = fit
        if(num_in_species > 0):
            self.adjusted_fitness = self.fitness / num_in_species  # local competition
            try:
                assert self.adjusted_fitness > - \
                    math.inf, f"adjusted fitness was -inf: fit: {self.fitness} n_in_species: {num_in_species}"
                assert self.adjusted_fitness < math.inf, f"adjusted fitness was -inf: fit: {self.fitness} n_in_species: {num_in_species}"
            except AssertionError as e:
                print(e)
        else:
            self.adjusted_fitness = self.fitness
            print("ERROR: num_in_species was 0")

    def eval_genetic_novelty(self, archive, k):
        """Find the average distance from this Network's genome to k nearest neighbors."""
        self.novelty = 0
        distances = [self.genetic_difference(solution) for solution in archive]
        distances.sort()  # shortest first
        closest_k = distances[0:k]
        average = np.average(closest_k, axis=0)
        if(average != average):
            average = 0
        self.novelty = average
        return average

    def enabled_connections(self):
        for c in self.connection_genome:
            if c.enabled:
                yield c

    def mutate_activations(self):
        eligible_nodes = list(self.hidden_nodes())
        if(c.output_activation is None):
            eligible_nodes.extend(self.output_nodes())
        if c.allow_input_activation_mutation:
            eligible_nodes.extend(self.input_nodes())
        for node in eligible_nodes:
            if(np.random.uniform(0, 1) < c.prob_mutate_activation):
                node.fn = choose_random_function(c)

    def mutate_weights(self):
        """ Each connection weight is perturbed with a fixed probability by
            adding a floating point number chosen from a uniform distribution of positive and negative values """
        weight_mutation_max = c.weight_mutation_max
        weight_mutation_probability = c.prob_mutate_weight
        
        for cx in self.connection_genome:
            if(np.random.uniform(0, 1) < weight_mutation_probability):
                cx.weight += np.random.uniform(-weight_mutation_max,
                                               weight_mutation_max)
            elif(np.random.uniform(0, 1) < c.prob_weight_reinit):
                cx.weight = self.random_weight()

        self.clamp_weights()  # TODO NOT SURE
        
    def mutate_weights_with_prob(self):
        """ Each connection weight is perturbed with a fixed probability by
            adding a floating point number chosen from a uniform distribution of positive and negative values """
        for cx in self.connection_genome:
            if(np.random.uniform(0, 1) < c.prob_mutate_weight):
                cx.weight += np.random.uniform(-c.weight_mutation_max,
                                               c.weight_mutation_max,)
            elif(np.random.uniform(0, 1) < c.prob_weight_reinit):
                cx.weight = self.random_weight()

        self.clamp_weights()  # TODO NOT SURE

    def mutate_random_weight(self, amount):
        try:
            cx = np.random.choice(self.connection_genome, 1)[
                0]  # choose one random connection
            cx.weight += amount
        except Exception as e:  # TODO no
            print(f"ERROR in mutation: {e}")

    def add_connection(self):
        chance_to_reenable = c.prob_reenable_connection
        allow_recurrent = c.allow_recurrent
        for i in range(20):  # try 20 times
            [fromNode, toNode] = np.random.choice(
                self.node_genome, 2, replace=False)
            existing_cx = None
            for cx in self.connection_genome:
                if cx.fromNode.uuid == fromNode.uuid and cx.toNode.uuid == toNode.uuid:
                    existing_cx = cx
                    break
            if(existing_cx != None):
                if(not existing_cx.enabled and np.random.rand() < chance_to_reenable):
                    existing_cx.enabled = True     # re-enable the connection
                break  # don't allow duplicates

            if(fromNode.layer == toNode.layer):
                continue  # don't allow two nodes on the same layer to connect

            is_recurrent = fromNode.layer > toNode.layer
            if(not allow_recurrent and is_recurrent):
                continue  # invalid

            # valid connection, add
            new_cx = Connection(fromNode, toNode, self.random_weight())
            self.connection_genome.append(new_cx)
            self.update_node_layers()
            break

        # failed to find a valid connection, don't add

    def disable_invalid_connections(self):
        to_remove = []
        for cx in self.connection_genome:
            if(cx.fromNode == cx.toNode):
                raise Exception("Nodes should not be self-recurrent")
            if(cx.toNode.layer == cx.fromNode.layer):
                to_remove.append(cx)
            cx.is_recurrent = cx.fromNode.layer > cx.toNode.layer
            if(not self.allow_recurrent and cx.is_recurrent):
                to_remove.append(cx)  # invalid TODO consider disabling instead

        for cx in to_remove:
            self.connection_genome.remove(cx)

    def add_node(self):
        eligible_cxs = [
            cx for cx in self.connection_genome if not cx.is_recurrent]
        if(len(eligible_cxs) < 1):
            return
        old = np.random.choice(eligible_cxs)
        new_node = Node(choose_random_function(c),
                        NodeType.Hidden, self.get_new_node_id())
        self.node_genome.append(new_node)  # add a new node between two nodes
        old.enabled = False  # disable old connection

        # The connection between the first node in the chain and the new node is given a weight of one
        # and the connection between the new node and the last node in the chain is given the same weight as the connection being split

        self.connection_genome.append(Connection(
            self.node_genome[old.fromNode.id], self.node_genome[new_node.id],   self.random_weight()))

        # TODO shouldn't be necessary
        self.connection_genome[-1].fromNode = self.node_genome[old.fromNode.id]
        self.connection_genome[-1].toNode = self.node_genome[new_node.id]
        self.connection_genome.append(Connection(
            self.node_genome[new_node.id],     self.node_genome[old.toNode.id], old.weight))

        self.connection_genome[-1].fromNode = self.node_genome[new_node.id]
        self.connection_genome[-1].toNode = self.node_genome[old.toNode.id]

        self.update_node_layers()
        # self.disable_invalid_connections() # TODO broken af

    def remove_node(self):
        # This is a bit of a buggy mess
        hidden = self.hidden_nodes()
        if(len(hidden) < 1):
            return
        node_id_to_remove = np.random.choice([n.id for n in hidden], 1)[0]
        for cx in self.connection_genome[::-1]:
            if(cx.fromNode.id == node_id_to_remove or cx.toNode.id == node_id_to_remove):
                self.connection_genome.remove(cx)
        for node in self.node_genome[::-1]:
            if node.id == node_id_to_remove:
                self.node_genome.remove(node)
                break

        self.update_node_layers()
        self.disable_invalid_connections()

    def disable_connection(self):
        eligible_cxs = list(self.enabled_connections())
        if(len(eligible_cxs) < 1):
            return
        cx = np.random.choice(eligible_cxs)
        cx.enabled = False

    def update_node_layers(self) -> int:
        # layer = number of edges in longest path between this node and input
        def get_node_to_input_len(current_node, current_path=0, longest_path=0, attempts=0):
            if(attempts > 1000):
                print("ERROR: infinite recursion while updating node layers")
                return longest_path
            # use recursion to find longest path
            if(current_node.type == NodeType.Input):
                return current_path
            all_inputs = [
                cx for cx in self.connection_genome if not cx.is_recurrent and cx.toNode.id == current_node.id]
            for inp_cx in all_inputs:
                this_len = get_node_to_input_len(
                    inp_cx.fromNode, current_path+1, attempts+1)
                if(this_len >= longest_path):
                    longest_path = this_len
            return longest_path

        highest_hidden_layer = 1
        for node in self.hidden_nodes():
            node.layer = get_node_to_input_len(node)
            highest_hidden_layer = max(node.layer, highest_hidden_layer)

        for node in self.output_nodes():
            node.layer = highest_hidden_layer+1

    def genetic_difference(self, other) -> float:
        # only enabled connections, sorted by innovation id
        this_cxs = sorted(self.enabled_connections(),
                          key=lambda c: c.innovation)
        other_cxs = sorted(other.enabled_connections(),
                           key=lambda c: c.innovation)

        N = max(len(this_cxs), len(other_cxs))
        other_innovation = [c.innovation for c in other_cxs]

        # number of excess connections
        n_excess = len(get_excess_connections(this_cxs, other_innovation))
        # number of disjoint connections
        n_disjoint = len(get_disjoint_connections(this_cxs, other_innovation))

        # matching connections
        this_matching, other_matching = get_matching_connections(
            this_cxs, other_cxs)
        difference_of_matching_weights = [
            abs(o_cx.weight-t_cx.weight) for o_cx, t_cx in zip(other_matching, this_matching)]
        if(len(difference_of_matching_weights) == 0):
            difference_of_matching_weights = 0
        difference_of_matching_weights = np.mean(
            difference_of_matching_weights)

        # Furthermore, the compatibility distance function
        # includes an additional argument that counts how many
        # activation functions differ between the two individuals
        n_different_fns = 0
        for t_node, o_node in zip(self.node_genome, other.node_genome):
            if(t_node.fn.__name__ != o_node.fn.__name__):
                n_different_fns += 1

        # can normalize by size of network (from Ken's paper)
        if(N > 0):
            n_excess /= N
            n_disjoint /= N

        # weight (values from Ken)
        n_excess *= 1
        n_disjoint *= 1
        difference_of_matching_weights *= .4
        n_different_fns *= 1
        difference = n_excess + n_disjoint + \
            difference_of_matching_weights + n_different_fns

        return difference

    def species_comparision(self, other, threshold) -> bool:
        # returns whether other is the same species as self
        return self.genetic_difference(other) <= threshold  # TODO equal to?

    def input_nodes(self) -> list:
        return self.node_genome[0:self.n_inputs]

    def output_nodes(self) -> list:
        return self.node_genome[self.n_inputs:self.n_inputs+self.n_outputs]

    def hidden_nodes(self) -> list:
        return self.node_genome[self.n_inputs+self.n_outputs:]

    def set_inputs(self, inputs):
        if(self.use_input_bias):
            inputs.append(1.0)  # bias = 1.0
        assert len(inputs) == self.n_inputs, f"Inputs must be of length {self.n_inputs}"
        for i, inp in enumerate(inputs):
            # inputs are first N nodes
            self.node_genome[i].sum_input = inp
            self.node_genome[i].output = self.node_genome[i].fn(inp)

    def get_layer(self, layer_index):
        for node in self.node_genome:
            if node.layer == layer_index:
                yield node

    def get_hidden_and_output_layers(self):
        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer
        return [self.get_layer(i) for i in range(1, output_layer+1)]

    def count_layers(self):
        return len(np.unique([node.layer for node in self.node_genome]))

    def clamp_weights(self):
        for cx in self.connection_genome:
            if(cx.weight < self.weight_threshold and cx.weight > -self.weight_threshold):
                cx.weight = 0
            if(cx.weight > self.max_weight):
                cx.weight = self.max_weight
            if(cx.weight < -self.max_weight):
                cx.weight = -self.max_weight

    def eval(self, inputs):
        self.set_inputs(inputs)
        return self.feed_forward()

    def feed_forward(self):
        if self.allow_recurrent:
            for node in self.get_layer(0):  # input nodes (handle recurrent)
                for node_input in list(filter(lambda x: x.toNode.id == node.id, self.enabled_connections())):
                    node.sum_input += node_input.fromNode.output * node_input.weight
                node.output = node.fn(node.sum_input)

        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer

        for layer_index in range(1, output_layer+1):
            # hidden and output layers:
            layer = self.get_layer(layer_index)
            for node in layer:
                node.sum_input = 0
                node.output = 0
                node_inputs = list(
                    filter(lambda x: x.toNode.id == node.id, self.enabled_connections()))  # cxs that end here
                for cx in node_inputs:
                    node.sum_input += cx.fromNode.output * cx.weight

                node.output = node.fn(node.sum_input)  # apply activation

        return [node.output for node in self.output_nodes()]

    def eval_substrate(self, res_h, res_w, color_mode):
        if self.allow_recurrent:
            raise Exception("Fast method doesn't work with recurrent yet")

        if Genome.pixel_inputs is None or Genome.pixel_inputs.shape[0] != res_h or Genome.pixel_inputs.shape[1] != res_w:
            # lazy init:
            x_vals = np.linspace(-.5, .5, res_w)
            y_vals = np.linspace(-.5, .5, res_h)
            Genome.pixel_inputs = np.zeros(
                (res_h, res_w, c.num_sensor_neurons), dtype=np.float32)
            for y in range(res_h):
                for x in range(res_w):
                    this_pixel = [y_vals[y], x_vals[x]]  # coordinates
                    if(self.use_input_bias):
                        this_pixel.append(1.0)  # bias = 1.0
                    Genome.pixel_inputs[y][x] = this_pixel

        for i in range(len(self.node_genome)):
            # initialize outputs to 0:
            self.node_genome[i].outputs = np.zeros((res_h, res_w))

        for i in range(c.num_sensor_neurons):
            # inputs are first N nodes
            self.node_genome[i].sum_inputs = Genome.pixel_inputs[:, :, i]
            self.node_genome[i].outputs = self.node_genome[i].fn(
                Genome.pixel_inputs[:, :, i])

        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer

        for layer_index in range(1, output_layer+1):
            # hidden and output layers:
            layer = self.get_layer(layer_index)
            for node in layer:
                node_inputs = list(
                    filter(lambda x: x.toNode.id == node.id, self.enabled_connections()))  # cxs that end here

                node.sum_inputs = np.zeros((res_h, res_w), dtype=np.float32)
                for cx in node_inputs:
                    if(not hasattr(cx.fromNode, "outputs")):
                        print(cx.fromNode.type)
                        print(list(self.enabled_connections()))
                        print(cx.fromNode)
                        print(self.node_genome)
                    inputs = cx.fromNode.outputs * cx.weight
                    node.sum_inputs = node.sum_inputs + inputs

                if(np.isnan(node.sum_inputs).any() or np.isinf(np.abs(node.sum_inputs)).any()):
                    print(f"inputs was {node.sum_inputs}")
                    node.sum_inputs = np.zeros(
                        (res_h, res_w), dtype=np.float32)  # TODO why?
                    node.outputs = node.sum_inputs  # ignore node

                node.outputs = node.fn(node.sum_inputs)  # apply activation
                node.outputs = node.outputs.reshape((res_h, res_w))
                # TODO not sure (SLOW)
                node.outputs = np.clip(node.outputs, -1, 1)

        outputs = [node.outputs for node in self.output_nodes()]
        if(color_mode == 'RGB' or color_mode == "HSL"):
            outputs = np.array(outputs).transpose(
                1, 2, 0)  # move color axis to end
        else:
            outputs = np.reshape(outputs, (res_h, res_w))
        self.image = outputs
        return outputs

    def reset_activations(self):
        for node in self.node_genome:
            node.outputs = np.zeros(
                (c.train_image.shape[0], c.train_image.shape[1]))
            node.sum_inputs = np.zeros(
                (c.train_image.shape[0], c.train_image.shape[1]))

    def save(self, filename):
        json_nodes = [(node.fn.__name__, node.type)
                      for node in self.node_genome]
        json_cxs = [(self.node_genome.index(cx.fromNode), self.node_genome.index(
            cx.toNode), cx.weight, cx.enabled) for cx in self.connection_genome]
        print(json_cxs)
        json_config = json.loads(c.to_json())
        with open(filename, 'w') as f:
            json.dump({'nodes': json_nodes, 'cxs': json_cxs,
                      "config": json_config}, f)
            f.close()

        c.from_json(json_config)


def crossover(parent1, parent2):
    [fit_parent, less_fit_parent] = sorted(
        [parent1, parent2], key=lambda x: x.fitness, reverse=True)
    # child = copy.deepcopy(fit_parent)
    child = Genome()
    child.species_id = fit_parent.species_id
    # disjoint/excess genes are inherited from more fit parent
    child.node_genome = copy.deepcopy(fit_parent.node_genome)
    child.connection_genome = copy.deepcopy(fit_parent.connection_genome)

    # child.more_fit_parent = fit_parent # TODO

    child.connection_genome.sort(key=lambda x: x.innovation)
    matching1, matching2 = get_matching_connections(
        fit_parent.connection_genome, less_fit_parent.connection_genome)
    for match_index in range(len(matching1)):
        # Matching genes are inherited randomly
        child_cx = child.connection_genome[[x.innovation for x in child.connection_genome].index(
            matching1[match_index].innovation)]
        child_cx.weight = \
            matching1[match_index].weight if np.random.rand(
            ) < .5 else matching2[match_index].weight

        new_from = copy.deepcopy(matching1[match_index].fromNode if np.random.rand(
        ) < .5 else matching2[match_index].fromNode)
        child_cx.fromNode = new_from
        if new_from.id<len(child.node_genome)-1:
            child.node_genome[new_from.id] = new_from
        else:
            continue # TODO

        new_to = copy.deepcopy(matching1[match_index].toNode if np.random.rand(
        ) < .5 else matching2[match_index].toNode)
        child_cx.toNode = new_to
        if new_to.id<len(child.node_genome)-1:
            child.node_genome[new_to.id] = new_to
        else:
            continue # TODO

        if(not matching1[match_index].enabled or not matching2[match_index].enabled):
            if(np.random.rand() < 0.75):  # from Stanley/Miikulainen 2007
                child.connection_genome[match_index].enabled = False

    for cx in child.connection_genome:
        # TODO this shouldn't be necessary
        cx.fromNode = child.node_genome[cx.fromNode.id]
        cx.toNode = child.node_genome[cx.toNode.id]
    child.update_node_layers()
    child.disable_invalid_connections()
    return child


if __name__ == "__main__":
    network = Genome()
    for i in range(2):
        network.add_node()
        network.add_node()
        network.add_connection()
    print(network.eval([1, 2, 3, 4]))
    visualize_network(network)
    
    