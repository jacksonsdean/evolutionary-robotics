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
from util import generate_body, generate_brain, visualize_network
import pybullet as p

from util import choose_random_function, visualize_network

class NodeType(IntEnum):
    Input = 0
    Output = 1
    Hidden = 2
    CPG = 3

class Node:
    current_id = c.num_sensor_neurons + c.num_motor_neurons
    
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
    
    def next_id():
        Node.current_id+=1
        return Node.current_id

class Connection:
    # connection            e.g.  2->5,  1->4
    # innovation_number            0      1
    # where innovation number is the same for all of same connection
    # i.e. 2->5 and 2->5 have same innovation number, regardless of Network
    innovations = [] # WRONG
    current_innovation = -1

    def get_innovation_wrong(toNode, fromNode):
        cx = (fromNode.id, toNode.id) # based on id
        # cx = (fromNode.fn.__name__, toNode.fn.__name__) # based on fn
        if(cx in Connection.innovations):
            return Connection.innovations.index(cx)
        else:
            Connection.innovations.append(cx)
            return len(Connection.innovations) - 1
        
    def get_innovation():
        # TODO correct:
        Connection.current_innovation += 1
        return Connection.current_innovation

    def __init__(self, fromNode, toNode, weight, enabled=True) -> None:
        self.fromNode = fromNode  # TODO change to node ids?
        self.toNode = toNode
        self.weight = weight
        # self.innovation = Connection.get_innovation()
        self.innovation = Connection.get_innovation_wrong(toNode,fromNode)
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
            
    # TODO delete:
    # return sorted([c1 for c1 in cxs_1 if (c1.innovation in [c2.innovation for c2 in cxs_2] and c1.fromNode.id in [c2.fromNode.id for c2 in cxs_2] and c1.toNode.id in [c2.toNode.id for c2 in cxs_2])], key=lambda x: x.innovation),\
        # sorted([c2 for c2 in cxs_2 if c2.innovation in [
            #    c1.innovation for c1 in cxs_1]], key=lambda x: x.innovation)


def find_node_with_id(nodes, id):
    for node in nodes:
        if node.id == id:
            return node
    return None
    

class Genome:
    pixel_inputs = None
    current_id = 0
    
    def get_id():
        output = Genome.current_id
        Genome.current_id+=1
        return output
       
    def __init__(self) -> None:
        self.set_initial_values()
        self.create_cppn(c.num_sensor_neurons, c.num_motor_neurons, c.hidden_nodes_at_start)

    def set_initial_values(self):
        self.fitness = -math.inf
        self.novelty = -math.inf
        self.adjusted_fitness = -math.inf
        self.species_id = -1
        self.image = None
        self.node_genome = []  # inputs first, then outputs, then hidden
        self.connection_genome = []
        self.id = Genome.get_id()
        self.bodyID = -1
        self.allow_recurrent = c.allow_recurrent
        self.use_input_bias = c.use_input_bias
        self.more_fit_parent = None  # for record-keeping
        self.max_weight = c.max_weight
        self.weight_threshold = c.weight_threshold

    def create_cppn(self, num_inputs, num_outputs, hidden_nodes_at_start):
        self.n_hidden_nodes = hidden_nodes_at_start
        self.n_inputs = num_inputs
        self.n_outputs = num_outputs
        total_node_count = num_inputs + \
            num_outputs + hidden_nodes_at_start
        for i in range(num_inputs):
            self.node_genome.append(
                Node(choose_random_function(), NodeType.Input, i, 0))
        
        if c.use_cpg:
           self.node_genome[-1].type = NodeType.CPG
        
        for i in range(num_inputs, num_inputs + num_outputs):
            output_fn = choose_random_function() if c.output_activation is None else c.output_activation
            self.node_genome.append(
                Node(output_fn, NodeType.Output, len(self.node_genome), 2))
            
        for i in range(num_inputs + num_outputs, total_node_count):
            self.node_genome.append(Node(choose_random_function(), NodeType.Hidden, self.get_new_node_id(), 1))

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

    def start_simulation(self, headless, show_debug_output=False, save_as_best=False):
        generate_body(self.id)
        generate_brain(self.id, self.node_genome, self.connection_genome)
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

    def random_weight(self):
        return np.random.uniform(-self.max_weight, self.max_weight)

    def get_new_node_id(self):
        # TODO wrong but works better
        new_id = 0
        while len(self.node_genome) > 0 and new_id in [node.id for node in self.node_genome]:
            new_id += 1
        return new_id
        # return Node.next_id() # right but not good

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
                node.fn = choose_random_function()

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

    # def add_node(self):
    #     try:
    #         eligible_cxs = [
    #             cx for cx in self.connection_genome if not cx.is_recurrent]
    #         if(len(eligible_cxs) < 1):
    #             return
    #         old = np.random.choice(eligible_cxs)
    #         new_node = Node(choose_random_function(),
    #                         NodeType.Hidden, self.get_new_node_id())
    #         self.node_genome.append(new_node)  # add a new node between two nodes
    #         old.enabled = False  # disable old connection

    #         # The connection between the first node in the chain and the new node is given a weight of one
    #         # and the connection between the new node and the last node in the chain is given the same weight as the connection being split

    #         self.connection_genome.append(Connection(
    #             find_node_with_id(self.node_genome, old.fromNode.id), self.node_genome[-1],   self.random_weight()))
    #         # self.connection_genome.append(Connection(
    #         #    self.node_genome[-1], find_node_with_id(self.node_genome, old.toNode.id), self.random_weight()))

    #         # TODO shouldn't be necessary
    #         self.connection_genome[-1].fromNode = find_node_with_id(self.node_genome, old.fromNode.id)
    #         self.connection_genome[-1].toNode = new_node
    #         self.connection_genome.append(Connection(
    #             self.node_genome[new_node.id],     find_node_with_id(self.node_genome, old.toNode.id), old.weight))

    #         self.connection_genome[-1].fromNode = find_node_with_id(self.node_genome, new_node.id)
    #         self.connection_genome[-1].toNode = find_node_with_id(self.node_genome, old.toNode.id)

    #         self.update_node_layers()
    #         # self.disable_invalid_connections() # TODO broken af
    #     except Exception as e:
    #         print(f"ERROR in add_node: {e}")
    #         return # TODO
    def add_node(self):
        eligible_cxs = [
            cx for cx in self.connection_genome if not cx.is_recurrent]
        if(len(eligible_cxs) < 1):
            return
        old = np.random.choice(eligible_cxs)
        new_node = Node(choose_random_function(),
                        NodeType.Hidden, self.get_new_node_id())
        self.node_genome.append(new_node)  # add a new node between two nodes
        old.enabled = False  # disable old connection

        # The connection between the first node in the chain and the new node is given a weight of one
        # and the connection between the new node and the last node in the chain is given the same weight as the connection being split

        # self.connection_genome.append(Connection(
            # self.node_genome[old.fromNode.id], self.node_genome[new_node.id],   self.random_weight()))
        self.connection_genome.append(Connection(
                find_node_with_id(self.node_genome, old.fromNode.id), self.node_genome[-1],   self.random_weight()))

        # TODO shouldn't be necessary
        self.connection_genome[-1].fromNode = find_node_with_id(self.node_genome, old.fromNode.id)
        self.connection_genome[-1].toNode = find_node_with_id(self.node_genome, new_node.id)
        self.connection_genome.append(Connection(
            find_node_with_id(self.node_genome, new_node.id),     find_node_with_id(self.node_genome,old.toNode.id), old.weight))

        self.connection_genome[-1].fromNode = find_node_with_id(self.node_genome, new_node.id)
        self.connection_genome[-1].toNode = find_node_with_id(self.node_genome, old.toNode.id)

        self.update_node_layers()
        self.disable_invalid_connections()
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

if __name__ == "__main__":
    network = Genome()
    for i in range(2):
        network.add_node()
        network.add_node()
        network.add_connection()
    print(network.eval([1, 2, 3, 4]))
    visualize_network(network)
    
    