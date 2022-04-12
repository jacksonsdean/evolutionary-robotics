import math
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange
from activations import tanh
from neat_genome import Connection, Genome, NodeType, Node, find_node_with_id, get_matching_connections
from util import choose_random_function, generate_body, generate_brain, get_avg_number_of_connections, get_avg_number_of_hidden_nodes, get_max_number_of_connections, get_max_number_of_hidden_nodes, visualize_hn_phenotype_network, visualize_network
from species import *
import copy 
from neat import NEAT, classic_selection_and_reproduction, update_solution_archive
import random
import copy
import os
import platform
import constants as c


class Substrate:
    def __init__(self):
        layer_heights = [c.num_sensor_neurons]
        layer_heights.extend([c.num_hn_hidden_nodes_per_layer]*c.num_hn_hidden_layers)
        layer_heights.append(c.num_motor_neurons)
        total_num_layers = len(layer_heights)
        self.x = np.linspace(-1,1,total_num_layers)
        self.y = []
        for h in layer_heights:
            self.y.append(list(np.linspace(-1,1,h)))

    def visualize(self):
        xs = []
        ys = []
        for x in s.x:
            for y in s.y:
                xs.append(x)
                ys.append(y)
        data = np.vstack([xs, ys])
        plt.imshow(data, vmin=-1, vmax=1)
        plt.show()

class HyperNEAT(NEAT):
    def evolve(self):
            for i in range(c.pop_size): # only create parents for initialization (the mu in mu+lambda)
                self.population.append(HyperNEATGenome()) # generate new random individuals as parents
                
            if c.use_speciation:
                assign_species(self.all_species, self.population, self.species_threshold, Species)

            # Run NEAT
            for self.gen in range(c.num_gens):
                self.run_one_generation()
    
    def mutate(self, child, rates):
        prob_mutate_activation, prob_mutate_weight, prob_add_connection, prob_add_node, prob_remove_node, prob_disable_connection, weight_mutation_max, prob_reenable_connection = rates
        child.fitness, child.adjusted_fitness = -math.inf, -math.inf # new fitnesses after mutation

        if(np.random.uniform(0,1) < c.prob_random_restart):
            child = HyperNEATGenome()
        if(np.random.uniform(0,1) < prob_add_node):
            child.add_node()
        if(np.random.uniform(0,1) < prob_remove_node):
            child.remove_node()
        if(np.random.uniform(0,1) < prob_add_connection):
            child.add_connection()
        if(np.random.uniform(0,1) < prob_disable_connection):
            child.disable_connection()
        # if(np.random.uniform(0,1)< prob_mutate_activation):
        
        child.mutate_activations()
        child.mutate_weights()
    
    def crossover(self, parent1, parent2):
        [fit_parent, less_fit_parent] = sorted(
            [parent1, parent2], key=lambda x: x.fitness, reverse=True)
        # child = copy.deepcopy(fit_parent)
        child = HyperNEATGenome()
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
            inherit_from_more_fit = np.random.rand() < .5 
            
            child_cx = child.connection_genome[[x.innovation for x in child.connection_genome].index(
                matching1[match_index].innovation)]
            child_cx.weight = \
                matching1[match_index].weight if inherit_from_more_fit else matching2[match_index].weight

            new_from = copy.deepcopy(matching1[match_index].fromNode if inherit_from_more_fit else matching2[match_index].fromNode)
            child_cx.fromNode = new_from
            # if new_from.id<len(child.node_genome):
            existing = find_node_with_id(child.node_genome, new_from.id)
            index_existing = child.node_genome.index(existing)
            child.node_genome[index_existing] = new_from
            # else:
                # print("********ERR:new from id", new_from.id, "len:", len(child.node_genome))
                # continue # TODO

            new_to = copy.deepcopy(matching1[match_index].toNode if inherit_from_more_fit else matching2[match_index].toNode)
            child_cx.toNode = new_to
            # if new_to.id<len(child.node_genome):
            existing = find_node_with_id(child.node_genome, new_to.id)
            index_existing = child.node_genome.index(existing)
            child.node_genome[index_existing] = new_to
            # else:
                # print("********ERR: new to id", new_to.id, "len:", len(child.node_genome))
                # continue # TODO

            if(not matching1[match_index].enabled or not matching2[match_index].enabled):
                if(np.random.rand() < 0.75):  # from Stanley/Miikulainen 2007
                    child.connection_genome[match_index].enabled = False

        for cx in child.connection_genome:
            cx.fromNode = find_node_with_id(child.node_genome, cx.fromNode.id)
            cx.toNode = find_node_with_id(child.node_genome, cx.toNode.id)
            assert cx.fromNode in child.node_genome, f"{child.id}: {cx.fromNode.id} {child.node_genome[cx.fromNode.id].id}"
            assert cx.toNode in child.node_genome, f"{child.id}: {cx.toNode.id} {child.node_genome[cx.toNode.id].id}"
            # TODO this shouldn't be necessary
            
        child.update_node_layers()
        # child.disable_invalid_connections()
        
        return child

    def save_best_network_image(self):
        best = self.get_best()
        visualize_network(self.get_best(), sample=False, save_name=f"best/{time.time()}_{self.gen}_{best.id}.png", extra_text="Generation: " + str(self.gen) + " fit: " + str(best.fitness) + " species: " + str(best.species_id))
        best.save_network_phenotype_image(self.gen, best.fitness, best.species_id)



class HyperNEATGenome(Genome):
    def __init__(self, **kwargs):
        self.set_initial_values()
        self.substrate = Substrate()
        self.create_cppn(c.num_hn_inputs, c.num_hn_outputs, c.hidden_nodes_at_start)
        self.phenotype_nodes = []
        
        
        for i in range(c.num_sensor_neurons):
            self.phenotype_nodes.append(Node(tanh, NodeType.Input, i, 0))

        for j in range(c.num_motor_neurons):
            self.phenotype_nodes.append(Node(tanh,NodeType.Output, j+c.num_sensor_neurons, 1+c.num_hn_hidden_layers))

        for i in range(c.num_hn_hidden_layers):
            self.phenotype_nodes.extend([Node(tanh, NodeType.Hidden, j+c.num_sensor_neurons+c.num_motor_neurons, i+1) for j in range(c.num_hn_hidden_nodes_per_layer)])

        # set x and y values:
        for node in self.phenotype_nodes:
            layer_nodes = [n for n in self.phenotype_nodes if n.layer == node.layer]
            num_in_layer = len(layer_nodes)
            index_in_layer = layer_nodes.index(node)
            
            node.x = np.linspace(-1, 1, c.num_hn_hidden_layers+2)[node.layer]
            node.y = np.linspace(-1, 1, num_in_layer)[index_in_layer]
            node.outputs = 0
            node.sum_inputs = 0

        
        self.phenotype_connections = []
        for node0 in self.phenotype_nodes:
            for node1 in self.phenotype_nodes:
                if not node0.layer >= node1.layer:
                    # no recurrent for now
                    self.phenotype_connections.append(Connection(node0, node1, 0)) 

  
    def start_simulation(self, headless, show_debug_output=False, save_as_best=False):
        self.eval_substrate_simple()
        generate_body(self.id)
        generate_brain(self.id, self.phenotype_nodes, self.phenotype_connections)
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
            
    def save_network_phenotype_image(self, gen, fitness, species_id):
        visualize_hn_phenotype_network(self.phenotype_connections,self.phenotype_nodes, sample=False, save_name=f"hyperneat_phenotypes/{time.time()}_{gen}_{self.id}.png", extra_text="Generation: " + str(gen) + " fit: " + str(fitness) + " species: " + str(species_id))

    def visualize_phenotype_network(self):
        visualize_hn_phenotype_network(self.phenotype_connections,self.phenotype_nodes)

    def create_output_visualization(self, res_h, res_w):
        output = np.zeros((res_h, res_w, 1), dtype=np.float32)
        for x1 in range(res_w):
            for x2 in range(res_w):
                for y2 in range(res_h):
                    for y1 in range(res_h):
                        for i in range(len(self.node_genome)):
                            # initialize outputs to 0:
                            self.node_genome[i].outputs = 0
                        coord_inputs = [(2*x1/res_h)-1., (2*y1/res_h)-1., (2*x2/res_w)-1., (2*y2/res_h)-1.]
                        for i in range(self.n_inputs):
                            # inputs are first N nodes
                            self.node_genome[i].sum_inputs = coord_inputs[i]
                            self.node_genome[i].outputs = self.node_genome[i].fn(coord_inputs[i])

                        # always an output node
                        output_layer = self.node_genome[self.n_inputs].layer

                        for layer_index in range(1, output_layer+1):
                            # hidden and output layers:
                            layer = self.get_layer(layer_index)
                            for node in layer:
                                node_inputs = list(
                                    filter(lambda x: x.toNode.id == node.id, self.enabled_connections()))  # cxs that end here

                                node.sum_inputs = 0
                                for cx in node_inputs:
                                    inputs = cx.fromNode.outputs * cx.weight
                                    node.sum_inputs = node.sum_inputs + inputs

                                node.outputs = node.fn(node.sum_inputs)  # apply activation

                        weight = [node.outputs for node in self.output_nodes()]
                        # output[int((y1+y2)/2.), int((x1+x2)/2.)] = weight
                        output[int((x1+x2)/2.), int((y1+y2)/2.)] = weight
        return output
    def eval_substrate_simple(self):
        """ Calculates the weights of connections by passing the substrate through the CPPN
        Input:  (x1, y1, x2, y2)
        Output: (weight)
        """
        for phen_cx in self.phenotype_connections:
            x1 = phen_cx.fromNode.x
            y1 = phen_cx.toNode.y
            x2 = phen_cx.toNode.x
            y2 = phen_cx.fromNode.y
            for i in range(len(self.node_genome)):
                # initialize outputs to 0:
                self.node_genome[i].outputs = 0
            coord_inputs = [x1, y1, x2, y2]
            for i in range(self.n_inputs):
                # inputs are first N nodes
                self.node_genome[i].sum_inputs = coord_inputs[i]
                self.node_genome[i].outputs = self.node_genome[i].fn(coord_inputs[i])

            # always an output node
            output_layer = self.node_genome[self.n_inputs].layer

            for layer_index in range(1, output_layer+1):
                # hidden and output layers:
                layer = self.get_layer(layer_index)
                for node in layer:
                    node_inputs = list(
                        filter(lambda x: x.toNode.id == node.id, self.enabled_connections()))  # cxs that end here

                    node.sum_inputs = 0
                    for cx in node_inputs:
                        inputs = cx.fromNode.outputs * cx.weight
                        node.sum_inputs = node.sum_inputs + inputs

                    node.outputs = node.fn(node.sum_inputs)  # apply activation

            weight = [node.outputs for node in self.output_nodes()]
            phen_cx.weight = weight[0]

            


if __name__ == "__main__":
    g = HyperNEATGenome()
    vis = g.create_output_visualization(32, 32)
    plt.imshow(vis, vmin=-1, vmax=1)
    plt.show()
    g.eval_substrate_simple()
    g.visualize_phenotype_network()    
    # for cx in g.phenotype_connections:
        # print(cx.fromNode.x, cx.fromNode.y, "->", cx.toNode.x, cx.toNode.y, cx.weight)
    # count = 0
    # z = np.zeros((len(g.phenotype_connections), len(g.phenotype_connections)))
    # x, y = [],[]
    # for i, node0 in enumerate(g.phenotype_nodes):
    #     for j, node1 in enumerate(g.phenotype_nodes):
    #         if not node0.layer <= node1.layer:
    #             x_loc = ( node0.x + node1.x ) / 2
    #             y_loc = ( node0.y + node1.y ) / 2
    #             x.append(x_loc)
    #             y.append(y_loc)
    #             z[i,j] = g.phenotype_connections[count].weight
    #             count+=1
    #         else:
    #             z[i,j] = 0
    # print(len(x), len(y), len(z))
    # plt.pcolor(x,y,z)
    # plt.show()

