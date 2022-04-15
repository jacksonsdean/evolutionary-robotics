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
        ...

    def visualize(self):
        ...
        xs = []
        ys = []
        for x in s.x:
            for y in s.y:
                xs.append(x)
                ys.append(y)
        data = np.vstack([xs, ys])
        plt.imshow(data, vmin=-1, vmax=1)
        plt.show()

    def assign_node_positions(self, nodes):
       raise NotImplementedError()
    def get_connections(self, nodes):
       raise NotImplementedError()
   
    def visualize_node_positions(self,nodes):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for node in nodes:
                ax.scatter(node.x, node.y, c='b')
                ax.text(node.x-.1, node.y+0.1, f"{node.layer}.{node.id}")
            plt.show()
            
    def visualize_substrate(self,nodes, connections):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for node in nodes:
            ax.scatter(node.x, node.y, c='b')
            ax.text(node.x-.1, node.y+0.1, f"{node.layer}.{node.id}")
            
        for connection in connections:
            ax.plot([connection.fromNode.x, connection.toNode.x], [connection.fromNode.y, connection.toNode.y], c='g', alpha=0.5)
        plt.show()

class GridSubstrate(Substrate):
    def assign_node_positions(self, nodes):
        x_space = np.linspace(1, -1, c.num_hn_hidden_layers+2)
        for node in nodes:
            layer_nodes = [n for n in nodes if n.layer == node.layer]
            num_in_layer = len(layer_nodes)
            y_space = np.linspace(1, -1, num_in_layer)
            index_in_layer = layer_nodes.index(node)
            node.x = x_space[node.layer]
            node.y = y_space[index_in_layer]
            node.outputs = 0
            node.sum_inputs = 0
            
    def get_connections(self, nodes):
        output = []
        for node0 in nodes:
            for node1 in nodes:
                # if not node0.layer >= node1.layer:
                # if not node0.layer != node1.layer:
                    output.append(Connection(node0, node1, 0)) 
        return output

class SandwichSubstrate(Substrate):
    sensor_layout = [
        [0, 0], # Touch FrontLowerLeg   
        [1, 0], # Touch BackLowerLeg  
        [2, 0], # Touch LeftLowerLeg  
        [3, 0], # Touch RightLowerLeg 

        [0, 1], # Rotation BackLegRot_BackLeg
        [1, 1], # Rotation FrontLegRot_FrontLeg
        [2, 1], # Rotation LeftLegRot_LeftLeg
        [3, 1], # Rotation LeftLegRot_LeftLeg

        [0, 2], # Rotation Torso_BackLegRot
        [1, 2], # Rotation Torso_FrontLegRot
        [2, 2], # Rotation Torso_LeftLegRot
        [3, 2], # Rotation Torso_RightLegRot

        [0, 3], # Rotation BackLeg_BackLowerLeg
        [1, 3], # Rotation FrontLeg_FrontLowerLeg
        [2, 3], # Rotation LeftLeg_LeftLowerLeg
        [3, 3], # Rotation RightLeg_RightLowerLeg
        
        [0, 4], # Torso velocity
    ]
    

    output_layout = [
        [0, 0], # Torso_BackLegRot
        [1, 0], # Torso_FrontLegRot
        [2, 0], # Torso_LeftLegRot
        [3, 0], # Torso_RightLegRot
        
        [0, 1], # BackLeg_BackLowerLeg
        [1, 1], # FrontLeg_FrontLowerLeg
        [2, 1], # LeftLeg_LeftLowerLeg
        [3, 1], # RightLeg_RightLowerLeg

        [0, 2], # BackLegRot_BackLeg
        [1, 2], # FrontLegRot_FrontLeg
        [2, 2], # LeftLegRot_LeftLeg
        [3, 2], # RightLegRot_RightLeg
    ]


    def __init__(self):
        
        SandwichSubstrate.sensor_layout = SandwichSubstrate.sensor_layout[:c.num_sensor_neurons]
        SandwichSubstrate.output_layout = SandwichSubstrate.output_layout[:c.num_motor_neurons]

        SandwichSubstrate.output_rows = np.max([s[1] for s in SandwichSubstrate.output_layout])+1
        SandwichSubstrate.output_cols = np.max([s[0] for s in SandwichSubstrate.output_layout])+1
        SandwichSubstrate.sensor_rows = np.max([s[1] for s in SandwichSubstrate.sensor_layout])+1
        SandwichSubstrate.sensor_cols = np.max([s[0] for s in SandwichSubstrate.sensor_layout])+1
        
        if isinstance(c.num_hn_hidden_nodes_per_layer, list):
            SandwichSubstrate.hidden_rows = [math.ceil(math.sqrt(l)) for l in c.num_hn_hidden_nodes_per_layer]
            SandwichSubstrate.hidden_cols = [math.ceil(math.sqrt(l)) for l in c.num_hn_hidden_nodes_per_layer]
        else:
            SandwichSubstrate.hidden_rows = math.ceil(math.sqrt(c.num_hn_hidden_nodes_per_layer))
            SandwichSubstrate.hidden_cols = math.ceil(math.sqrt(c.num_hn_hidden_nodes_per_layer))
      
        if c.use_cpg:
            SandwichSubstrate.sensor_layout.append([1, SandwichSubstrate.sensor_rows-1]), # CPG
            SandwichSubstrate.sensor_cols += 1

    def assign_node_positions(self, nodes):
        layers = []
        layer_counts = []
        for node in nodes:
            if node.layer not in layers:
                layers.append(node.layer)
            layer_counts.append(len([n for n in nodes if n.layer == node.layer]))

        for i, layer in enumerate(layers):
            rows, cols = 0, 0
            if i == 0:
                # input
                rows = self.sensor_rows
                cols = self.sensor_cols
            elif i == len(layers)-1:
                # output
                rows = self.output_rows
                cols = self.output_cols
            else:
                # hidden
                if isinstance(c.num_hn_hidden_nodes_per_layer, list):
                    rows = self.hidden_rows[i-1]
                    cols = self.hidden_cols[i-1]
                else:    
                    rows = self.hidden_rows
                    cols = self.hidden_cols
            x_space = np.linspace(1, -1, cols) if cols > 1 else [0] * cols
            y_space = np.linspace(1, -1, rows) if rows > 1 else [0] * rows
            index_in_layer = 0
            for node in nodes:
                if node.layer != i:
                    continue
                if i == 0:
                    # input
                    node.x = x_space[self.sensor_layout[index_in_layer][0]]
                    node.y = y_space[self.sensor_layout[index_in_layer][1]]
                elif i == len(layers)-1:
                    # output
                    node.x = x_space[self.output_layout[index_in_layer][0]]
                    node.y = y_space[self.output_layout[index_in_layer][1]]
                else:
                    # hidden
                    node.x = x_space[index_in_layer%cols]
                    node.y = y_space[math.floor(index_in_layer/cols)]
                node.outputs = 0
                node.sum_inputs = 0
                index_in_layer += 1
                
    def visualize_substrate(self,nodes, connections, weights=None):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        for node in nodes:
            ax.scatter(node.x, node.layer, node.y, c='b', marker='o', s=30)
            # ax.text(node.x-.1, node.y+0.1, node.layer+0.1, f"{node.layer}.{node.id}")
        ax.set_xlabel('X')
        ax.set_ylabel('Layer')
        ax.set_zlabel('Y')
        for i, connection in enumerate(connections):
            if weights is None:
                color = 'k'
                alpha = .5
            else:
                color = 'r' if weights[i] < 0 else 'k'
                alpha = abs(weights[i]) / c.max_weight
                

            ax.plot([connection.fromNode.x, connection.toNode.x],[connection.fromNode.layer, connection.toNode.layer], [connection.fromNode.y, connection.toNode.y], c=color, alpha=alpha)
        plt.tight_layout()
        plt.show()
        
    def get_connections(self, nodes):
        output = []
        layers = []
        for node in nodes:
            if node.layer not in layers:
                layers.append(node.layer)
        for layer in layers:
            for node0 in nodes:
                for node1 in nodes:
                    if node0.layer == layer and node1.layer == layer+1:
                        output.append(Connection(node0, node1, 0))
        return output

class HyperNEAT(NEAT):
    def evolve(self, run_number = 1, show_output= True):
        self.show_output = show_output or self.debug_output
        self.run_number = run_number
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

    def save_best_network_image(self, end_early=False):
        best = self.get_best()
        best.eval_substrate_simple()
        visualize_network(best, sample=False, save_name=f"best/{time.time()}_{self.gen}_{best.id}.png", extra_text="Generation: " + str(self.gen) + " fit: " + str(best.fitness) + " species: " + str(best.species_id))
        # plt.close()
        best.save_network_phenotype_image(self.gen, best.fitness, best.species_id)
        # plt.close()
        if self.gen == c.num_gens -1 or end_early:
            print("Saving weight map...")
            vis = best.create_output_visualization(32, 32)
            vis = vis.reshape(32,32)
            plt.imsave(f"hyperneat_phenotypes/{time.time()}_hyperneat_phenotype_vis.png", vis, vmin=-1, vmax=1)
            # plt.close()
            best.substrate.visualize_substrate(best.phenotype_nodes, best.phenotype_connections, best.weights)
        
class HyperNEATGenome(Genome):
    def __init__(self, **kwargs):
        self.set_initial_values()
        if c.substrate_type == "sandwich":
            self.substrate = SandwichSubstrate()
        elif c.substrate_type == "grid":
            self.substrate = GridSubstrate()
        self.create_cppn(c.num_hn_inputs, c.num_hn_outputs, c.hidden_nodes_at_start)
        self.phenotype_nodes = []
        
        
        for i in range(c.num_sensor_neurons):
            self.phenotype_nodes.append(Node(tanh, NodeType.Input, i, 0))

        for j in range(c.num_motor_neurons):
            self.phenotype_nodes.append(Node(tanh,NodeType.Output, j+c.num_sensor_neurons, 1+c.num_hn_hidden_layers))

        for i in range(c.num_hn_hidden_layers):
            if isinstance(c.num_hn_hidden_nodes_per_layer, list):
                self.phenotype_nodes.extend([Node(tanh, NodeType.Hidden, j+c.num_sensor_neurons+c.num_motor_neurons, i+1) for j in range(c.num_hn_hidden_nodes_per_layer[i])])
            else:
                self.phenotype_nodes.extend([Node(tanh, NodeType.Hidden, j+c.num_sensor_neurons+c.num_motor_neurons, i+1) for j in range(c.num_hn_hidden_nodes_per_layer)])

        # set x and y values:
        self.substrate.assign_node_positions(self.phenotype_nodes)
        self.phenotype_connections = self.substrate.get_connections(self.phenotype_nodes)
        self.weights = np.zeros(len(self.phenotype_connections))

        # self.substrate.visualize_substrate(self.phenotype_nodes, self.phenotype_connections)
  
    def start_simulation(self, headless, show_debug_output=False, save_as_best=False):
        # self.eval_substrate_fast()
        self.eval_substrate_simple()
        # self.substrate.visualize_substrate(self.phenotype_nodes, self.phenotype_connections, self.weights)
        
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
        visualize_hn_phenotype_network(self.phenotype_connections,self.phenotype_nodes,  visualize_disabled=True)
    def visualize_genotype_network(self):
        visualize_network(self, visualize_disabled=True)

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
                        output[int((y1+y2)/2.), int((x1+x2)/2.)] = weight
                        
        return output
    def eval_substrate_simple(self):
        """ Calculates the weights of connections by passing the substrate through the CPPN
        Input:  (x1, y1, x2, y2)
        Output: (weight)
        """
        for i in range(len(self.node_genome)):
            # initialize outputs to 0:
            self.node_genome[i].outputs = 0
        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer
        self.weights = np.zeros(len(self.phenotype_connections))
        for cx_index, phen_cx in enumerate(self.phenotype_connections):
            x1 = phen_cx.fromNode.x
            y1 = phen_cx.toNode.y
            x2 = phen_cx.toNode.x
            y2 = phen_cx.fromNode.y
            coord_inputs = [x1, y1, x2, y2]
            for i in range(self.n_inputs):
                # inputs are first N nodes
                self.node_genome[i].sum_inputs = coord_inputs[i]
                self.node_genome[i].outputs = self.node_genome[i].fn(coord_inputs[i])

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
            net_output = weight[0] if (weight[0] > c.weight_threshold or weight[0]< -c.weight_threshold) else 0.0
            self.weights[cx_index] = net_output * c.max_phen_weight
            phen_cx.weight = net_output * c.max_phen_weight
            
    def eval_substrate_fast(self):
        """ Calculates the weights of connections by passing the substrate through the CPPN
        Input:  (x1, y1, x2, y2)
        Output: (weight)
        """
        for i in range(len(self.node_genome)):
            # initialize outputs to 0:
            self.node_genome[i].outputs = 0
        # always an output node
        output_layer = self.node_genome[self.n_inputs].layer
        inputs = []
        for phen_cx in self.phenotype_connections:
            x1 = phen_cx.fromNode.x
            y1 = phen_cx.toNode.y
            x2 = phen_cx.toNode.x
            y2 = phen_cx.fromNode.y
            coord_inputs = [x1, y1, x2, y2]
            inputs.append(coord_inputs)
        inputs = np.array(inputs)
        for i in range(self.n_inputs):
            # inputs are first N nodes
            self.node_genome[i].sum_inputs = inputs[:,i]
            self.node_genome[i].outputs = self.node_genome[i].fn(inputs[:,i])

        for layer_index in range(1, output_layer+1):
            # hidden and output layers:
            layer = self.get_layer(layer_index)
            for node in layer:
                node_inputs = list(
                    filter(lambda x: x.toNode.id == node.id, self.enabled_connections()))  # cxs that end here

                node.sum_inputs = np.zeros(len(inputs))
                for cx in node_inputs:
                    inputs = cx.fromNode.outputs * cx.weight
                    node.sum_inputs = node.sum_inputs + inputs

                node.outputs = node.fn(node.sum_inputs)  # apply activation

        weights = [node.outputs for node in self.output_nodes()]
        weights = np.array(weights)
        weights= weights[0] # only one output node
        for i, phen_cx in enumerate(self.phenotype_connections):
            weight = weights[i]
            net_output = weight if (weight > c.weight_threshold or weight< -c.weight_threshold) else 0.0
            phen_cx.weight = net_output * c.max_phen_weight
            


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

