import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
# import pygraphviz as pgv 
# from node_functions import *
# from networkx.drawing.nx_agraph import graphviz_layout
import sys
import inspect
import random
import pyrosim.pyrosim as pyrosim
from scikits import bootstrap

import constants as c
import warnings
warnings.filterwarnings("ignore")
    
def choose_random_function():
    return random.choice(c.activations)


def name_to_fn(name):
    fns = inspect.getmembers(sys.modules["node_functions"])
    fns.extend([("", None)])
    def avg_pixel_distance_fitness():
            pass
    fns.extend([("avg_pixel_distance_fitness", avg_pixel_distance_fitness)])
    return fns[[f[0] for f in fns].index(name)][1]
    
def visualize_network(individual, sample_point=[.25]*c.num_sensor_neurons, color_mode="L", visualize_disabled=False, layout='multi', sample=False, show_weights=False, use_inp_bias=False, use_radial_distance=True, save_name=None, extra_text=None, curved=False):
    if(sample):
        individual.eval(sample_point)
        
    # nodes = individual.node_genome
    connections = individual.connection_genome

    max_weight = c.max_weight

    G = nx.DiGraph()
    function_colors = {}
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'darkviolet',
    #         'hotpink', 'chocolate', 'lawngreen', 'lightsteelblue']
    colors = ['lightsteelblue'] * len([node.fn for node in individual.node_genome])
    node_labels = {}

    node_size = 2000
    # plt.figure(figsize=(int(1+(individual.count_layers())*1.5), 6), frameon=False)
    # plt.figure(figsize=(7, 6), frameon=False)
    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=0, hspace=0)

    for i, fn in enumerate([node.fn for node in individual.node_genome]):
        function_colors[fn.__name__] = colors[i]
    function_colors["identity"] = colors[0]

    fixed_positions={}
    inputs = individual.input_nodes()
    
    for i, node in enumerate(inputs):
        G.add_node(node, color=function_colors[node.fn.__name__], shape='d', layer=(node.layer))
        if node.type == 0:
            node_labels[node] = f"S{i}:\n{node.fn.__name__}\n"+(f"{node.output:.3f}" if node.output!=None else "")
        else:
            node_labels[node] = f"CPG"
            
        fixed_positions[node] = (-4,((i+1)*2.)/len(inputs))

    for node in individual.hidden_nodes():
        G.add_node(node, color=function_colors[node.fn.__name__], shape='o', layer=(node.layer))
        node_labels[node] = f"{node.id}\n{node.fn.__name__}\n"+(f"{node.output:.3f}" if node.output!=None else "" )

    for i, node in enumerate(individual.output_nodes()):
        title = i
        G.add_node(node, color=function_colors[node.fn.__name__], shape='s', layer=(node.layer))
        node_labels[node] = f"M{title}:\n{node.fn.__name__}\n"+(f"{node.output:.3f}")
        fixed_positions[node] = (4, ((i+1)*2)/len(individual.output_nodes()))
    pos = {}
    # shells = [[node for node in individual.input_nodes()], [node for node in individual.hidden_nodes()], [node for node in individual.output_nodes()]]
    # pos=nx.shell_layout(G, shells, scale=2)
    # pos=nx.shell_layout(G, scale=2)
    # pos=nx.spectral_layout(G, scale=2)
    # pos=graphviz_layout(G, prog='neato') # neato, dot, twopi, circo, fdp, nop, wc, acyclic, gvpr, gvcolor, ccomps, sccmap, tred, sfdp, unflatten.
    fixed_nodes = fixed_positions.keys()
    if(layout=='multi'):
        pos=nx.multipartite_layout(G, scale=4,subset_key='layer')
    elif(layout=='spring'):
        pos=nx.spring_layout(G, scale=4)

    plt.figure(figsize=(10, 10))
    # pos = nx.shell_layout(G)
    # pos = fixed_positions
    # pos = nx.spring_layout(G, pos=pos, fixed=fixed_nodes,k=.1,  scale = 2, iterations=2000)
    # for f, p in fixed_positions.items():
    #     pos[f] = (p[0]*20, p[1]*20)
    shapes = set((node[1]["shape"] for node in G.nodes(data=True)))
    for shape in shapes:
        this_nodes = [sNode[0] for sNode in filter(
            lambda x: x[1]["shape"] == shape, G.nodes(data=True))]
        colors = [nx.get_node_attributes(G, 'color')[cNode] for cNode in this_nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors,
                            label=node_labels, node_shape=shape, nodelist=this_nodes)

    edge_labels = {}
    for cx in connections:
        if(not visualize_disabled and (not cx.enabled or np.isclose(cx.weight, 0))): continue
        style = ('-', 'k',  .5+abs(cx.weight)/max_weight) if cx.enabled else ('--', 'grey', .5+ abs(cx.weight)/max_weight)
        if(cx.enabled and cx.weight<0): style  = ('-', 'r', .5+abs(cx.weight)/max_weight)
        if cx.fromNode in G.nodes and cx.toNode in G.nodes:
            G.add_edge(cx.fromNode, cx.toNode, weight=f"{cx.weight:.4f}", pos=pos, style=style)
        else:
            print("Connection not in graph:", cx.fromNode.id, "->", cx.toNode.id)
        edge_labels[(cx.fromNode, cx.toNode)] = f"{cx.weight:.3f}"


    edge_colors = nx.get_edge_attributes(G,'color').values()
    edge_styles = shapes = set((s[2] for s in G.edges(data='style')))
    # use_curved = show_weights or individual.count_layers()<3
    use_curved = curved
    for s in edge_styles:
        edges = [e for e in filter(
            lambda x: x[2] == s, G.edges(data='style'))]
        nx.draw_networkx_edges(G, pos,
                                edgelist=edges,
                                arrowsize=25, arrows=True, 
                                node_size=[node_size]*1000,
                                style=s[0],
                                edge_color=[s[1]]*1000,
                                width =s[2],
                                connectionstyle= "arc3" if not use_curved else f"arc3,rad={0.2*random.random()}",
                                # connectionstyle= "arc3"
                            )
    
    if extra_text is not None:
        plt.text(0.5,0.05, extra_text, horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure)
        
    
    if (show_weights):
        nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=.75)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, format="PNG")
        # plt.close()
    else:
        plt.show()
        # plt.close()

def visualize_hn_phenotype_network(connection_genome, node_genome, sample_point=[.25]*c.num_sensor_neurons, visualize_disabled=False, layout='multi', sample=False, show_weights=False, use_inp_bias=False, use_radial_distance=True, save_name=None, extra_text=None):
    # nodes = individual.node_genome
    connections = connection_genome
    input_nodes = [n for n in node_genome if n.type == 0]
    output_nodes = [n for n in node_genome if n.type == 1]
    hidden_nodes = [n for n in node_genome if n.type == 2]
    max_weight = c.max_weight

    G = nx.DiGraph()
    function_colors = {}
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'darkviolet',
    #         'hotpink', 'chocolate', 'lawngreen', 'lightsteelblue']
    colors = ['lightsteelblue'] * len([node.fn for node in node_genome])
    node_labels = {}

    node_size = 2000
    # plt.figure(figsize=(int(1+(individual.count_layers())*1.5), 6), frameon=False)
    # plt.figure(figsize=(7, 6), frameon=False)
    plt.subplots_adjust(left=0, bottom=0, right=1.25, top=1.25, wspace=0, hspace=0)

    for i, fn in enumerate([node.fn for node in node_genome]):
        function_colors[fn.__name__] = colors[i]
    function_colors["identity"] = colors[0]

    fixed_positions={}
    inputs = input_nodes
    
    for i, node in enumerate(inputs):
        G.add_node(node, color=function_colors[node.fn.__name__], shape='d', layer=(node.layer))
        if node.type == 0:
            node_labels[node] = f"S{i}:\n{node.fn.__name__}\n"+(f"{node.output:.3f}" if node.output!=None else "")
        else:
            node_labels[node] = f"CPG"
            
        fixed_positions[node] = (-4,((i+1)*2.)/len(inputs))

    for node in hidden_nodes:
        G.add_node(node, color=function_colors[node.fn.__name__], shape='o', layer=(node.layer))
        node_labels[node] = f"{node.id}\n{node.fn.__name__}\n"+(f"{node.output:.3f}" if node.output!=None else "" )

    for i, node in enumerate(output_nodes):
        title = i
        G.add_node(node, color=function_colors[node.fn.__name__], shape='s', layer=(node.layer))
        node_labels[node] = f"M{title}:\n{node.fn.__name__}\n"+(f"{node.output:.3f}")
        fixed_positions[node] = (4, ((i+1)*2)/len(output_nodes))
    pos = {}
    # shells = [[node for node in individual.input_nodes()], [node for node in individual.hidden_nodes()], [node for node in individual.output_nodes()]]
    # pos=nx.shell_layout(G, shells, scale=2)
    # pos=nx.shell_layout(G, scale=2)
    # pos=nx.spectral_layout(G, scale=2)
    # pos=graphviz_layout(G, prog='neato') # neato, dot, twopi, circo, fdp, nop, wc, acyclic, gvpr, gvcolor, ccomps, sccmap, tred, sfdp, unflatten.
    fixed_nodes = fixed_positions.keys()
    if(layout=='multi'):
        pos=nx.multipartite_layout(G, scale=4,subset_key='layer')
    elif(layout=='spring'):
        pos=nx.spring_layout(G, scale=4)

    plt.figure(figsize=(10, 10))
    # pos = nx.shell_layout(G)
    # pos = fixed_positions
    # pos = nx.spring_layout(G, pos=pos, fixed=fixed_nodes,k=.1,  scale = 2, iterations=2000)
    # for f, p in fixed_positions.items():
    #     pos[f] = (p[0]*20, p[1]*20)
    shapes = set((node[1]["shape"] for node in G.nodes(data=True)))
    for shape in shapes:
        this_nodes = [sNode[0] for sNode in filter(
            lambda x: x[1]["shape"] == shape, G.nodes(data=True))]
        colors = [nx.get_node_attributes(G, 'color')[cNode] for cNode in this_nodes]
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors,
                            label=node_labels, node_shape=shape, nodelist=this_nodes)

    edge_labels = {}
    for cx in connections:
        if(not visualize_disabled and (not cx.enabled or np.isclose(cx.weight, 0))): continue
        style = ('-', 'k',  .5+abs(cx.weight)/max_weight) if cx.enabled else ('--', 'grey', .5+ abs(cx.weight)/max_weight)
        if(cx.enabled and cx.weight<0): style  = ('-', 'r', .5+abs(cx.weight)/max_weight)
        if cx.fromNode in G.nodes and cx.toNode in G.nodes:
            G.add_edge(cx.fromNode, cx.toNode, weight=f"{cx.weight:.4f}", pos=pos, style=style)
        else:
            print("Connection not in graph:", cx.fromNode.id, "->", cx.toNode.id)
        edge_labels[(cx.fromNode, cx.toNode)] = f"{cx.weight:.3f}"


    edge_colors = nx.get_edge_attributes(G,'color').values()
    edge_styles = shapes = set((s[2] for s in G.edges(data='style')))
    # use_curved = show_weights or individual.count_layers()<3

    for s in edge_styles:
        edges = [e for e in filter(
            lambda x: x[2] == s, G.edges(data='style'))]
        nx.draw_networkx_edges(G, pos,
                                edgelist=edges,
                                arrowsize=25, arrows=True, 
                                node_size=[node_size]*1000,
                                style=s[0],
                                edge_color=[s[1]]*1000,
                                width =s[2],
                                # connectionstyle= "arc3" if use_curved else "arc3,rad=0.2"
                                connectionstyle= "arc3"
                            )
    
    if extra_text is not None:
        plt.text(0.5,0.05, extra_text, horizontalalignment='center', verticalalignment='center', transform=plt.gcf().transFigure)
        
    
    if (show_weights):
        nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=.75)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, format="PNG")
        # plt.close()
    else:
        plt.show()
        # plt.close()


    ""
    # labels = nx.get_edge_attributes(G,'weight')



def plot_mean_and_bootstrapped_ci_over_time(input_data = None, dataset=None, name = "change me", x_label = "change me", y_label="change me", y_limit = None, plot_bootstrap = True, show=False, title=None):
    """
    
    parameters: 
    input_data: (numpy array of shape (max_k, num_repitions)) solution metric to plot
    name: (string) name for legend
    x_label: (string) x axis label
    y_label: (string) y axis label
    
    returns:
    None
    """
    fig, ax = plt.subplots() # generate figure and axes
    input_data = [np.array(x) for x in input_data if isinstance(x, list)]
    input_data = np.array(input_data)
    if isinstance(name, str): name = [name]; input_data = [input_data]

    # for this_input_data, this_name in zip(input_data, name):
    for index, this_name in enumerate(name):
        # print("plotting",this_name, "with shape", dataset[index].shape)
        this_input_data = dataset[index]
        total_generations = this_input_data.shape[1]
        if(plot_bootstrap):
            boostrap_ci_generation_found = np.zeros((2,total_generations))
            for this_gen in range(total_generations):
                boostrap_ci_generation_found[:,this_gen] = bootstrap.ci(this_input_data[:,this_gen], np.nanmean, alpha=0.05)


        ax.plot(np.arange(total_generations), np.nanmean(this_input_data,axis=0), label = this_name) # plot the fitness over time
        if plot_bootstrap:
            ax.fill_between(np.arange(total_generations), boostrap_ci_generation_found[0,:], boostrap_ci_generation_found[1,:],alpha=0.3) # plot, and fill, the confidence interval for fitness over time
        ax.set_xlabel(x_label) # add axes labels
        ax.set_ylabel(y_label)
        if y_limit: ax.set_ylim(y_limit[0],y_limit[1])
        if title is not None:
            plt.title(title)
        else:
            plt.title(y_label)
        plt.legend(loc='best'); # add legend
        if show:
            plt.show() 

def get_best_solution_from_all_runs(results):
    best_fit = -math.inf
    best = None
    run_index = -1
    for i, run in enumerate(results):
        sorted_run = sorted(run, key = lambda x: x.fitness, reverse=True)
        run_best = sorted_run[0]
        if(run_best.fitness > best_fit):
            best_fit = run_best.fitness
            best = run_best
            run_index = i
    return best, run_index


def get_max_number_of_hidden_nodes(population):
    max = 0
    for g in population:
        if len(list(g.hidden_nodes()))> max:
            max = len(list(g.hidden_nodes()))
    return max

def get_avg_number_of_hidden_nodes(population):
    count = 0
    for g in population:
        count+=len(g.node_genome) - g.n_inputs - g.n_outputs
    return count/len(population)

def get_max_number_of_connections(population):
    max_count = 0
    for g in population:
        count = len(list(g.enabled_connections()))
        if(count > max_count):
            max_count = count
    return max_count

def get_min_number_of_connections(population):
    min_count = math.inf
    for g in population:
        count = len(list(g.enabled_connections())) 
        if(count < min_count):
            min_count = count
    return min_count

def get_avg_number_of_connections(population):
    count = 0
    for g in population:
        count+=len(list(g.enabled_connections()))
    return count/len(population)
    

def generate_body(id):
    pyrosim.Start_URDF(f"body{id}.urdf")
    pyrosim.Send_Cube(name="Torso", pos=[0, 0, 1], size=[1, 1, 1], mass=c.torso_weight)
    pyrosim.Send_Joint( name = "Torso_BackLegRot" , parent= "Torso" , child = "BackLegRot" , type = "revolute", position = [0, -0.5, 1.0], jointAxis = "0 1 0")
    pyrosim.Send_Joint( name = "BackLegRot_BackLeg" , parent= "BackLegRot" , child = "BackLeg" , type = "revolute" if c.num_motor_neurons > 9 else "fixed", position = [0, 0, 0], jointAxis = "1 0 0")
    
    pyrosim.Send_Cube(name="BackLegRot", pos=[0.0, -0.5, 0.0], size=[0,0,0], mass=0.0)
    pyrosim.Send_Cube(name="BackLeg", pos=[0.0, -0.5, 0.0], size=[.2, 1., .2], mass=1.0)
    pyrosim.Send_Joint( name = "Torso_FrontLegRot" , parent= "Torso" , child = "FrontLegRot" , type = "revolute", position = [0.0, 0.5, 1.0], jointAxis = "1 0 0")
    pyrosim.Send_Joint( name ="FrontLegRot_FrontLeg" , parent= "FrontLegRot" , child = "FrontLeg" , type = "revolute" if c.num_motor_neurons > 9 else "fixed", position = [0.0, 0.0, 0.0], jointAxis = "0 1 0")
    pyrosim.Send_Cube(name="FrontLegRot", pos=[0.0, 0.5, 0], size=[0,0,0], mass=0.0)
    pyrosim.Send_Cube(name="FrontLeg", pos=[0.0, 0.5, 0], size=[.2, 1., .2], mass=1.0)
    pyrosim.Send_Cube(name="LeftLeg", pos=[-0.5, 0.0, 0.0], size=[1.0, 0.2, 0.2], mass=1.0)
    pyrosim.Send_Cube(name="LeftLegRot", pos=[-0.5, 0.0, 0.0], size=[0,0,0], mass=0.0)
    pyrosim.Send_Joint( name = "Torso_LeftLegRot" , parent= "Torso" , child = "LeftLegRot" , type = "revolute", position = [-0.5, 0, 1.], jointAxis = "1 0 0")
    pyrosim.Send_Joint( name = "LeftLegRot_LeftLeg" , parent= "LeftLegRot" , child = "LeftLeg" , type = "revolute" if c.num_motor_neurons > 9 else "fixed", position = [0,0,0], jointAxis = "0 1 0" )
    pyrosim.Send_Cube(name="RightLegRot", pos=[0.5, 0.0, 0.0], size=[0,0,0], mass=0.0)
    pyrosim.Send_Cube(name="RightLeg", pos=[0.5, 0.0, 0.0], size=[1.0, 0.2, 0.2], mass=1.0)
    pyrosim.Send_Joint( name = "Torso_RightLegRot" , parent= "Torso" , child = "RightLegRot" , type = "revolute", position = [0.5, 0, 1.], jointAxis = "1 0 0")
    pyrosim.Send_Joint( name = "RightLegRot_RightLeg" , parent= "RightLegRot" , child = "RightLeg" , type = "revolute" if c.num_motor_neurons > 9 else "fixed", position = [0,0,0], jointAxis = "0 1 0" )
    pyrosim.Send_Cube(name="FrontLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.], mass=1.0)
    pyrosim.Send_Joint( name = "FrontLeg_FrontLowerLeg" , parent= "FrontLeg" , child = "FrontLowerLeg" , type = "revolute", position = [0,1,0], jointAxis = "1 0 0")
    pyrosim.Send_Cube(name="BackLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.], mass=1.0)
    pyrosim.Send_Joint( name = "BackLeg_BackLowerLeg" , parent= "BackLeg" , child = "BackLowerLeg" , type = "revolute", position = [0,-1,0], jointAxis = "1 0 0")
    pyrosim.Send_Cube(name="LeftLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.], mass=1.0)
    pyrosim.Send_Joint( name = "LeftLeg_LeftLowerLeg" , parent= "LeftLeg" , child = "LeftLowerLeg" , type = "revolute", position = [-1,0,0], jointAxis = "0 1 0")
    pyrosim.Send_Cube(name="RightLowerLeg", pos=[0.0, 0.0, -.5], size=[.2, .2, 1.], mass=1.0)
    pyrosim.Send_Joint( name = "RightLeg_RightLowerLeg" , parent= "RightLeg" , child = "RightLowerLeg" , type = "revolute", position = [1,0,0], jointAxis = "0 1 0")
    pyrosim.End()

def generate_brain(id, node_genome, connection_genome):
    pyrosim.Start_NeuralNetwork(f"brain{id}.nndf")

    # Neurons:
    # -Input
    n = 0
    pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "FrontLowerLeg", activation=node_genome[n].fn); n+=1
    pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "BackLowerLeg", activation=node_genome[n].fn); n+=1
    pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "LeftLowerLeg", activation=node_genome[n].fn); n+=1
    pyrosim.Send_Touch_Sensor_Neuron(name = n , linkName = "RightLowerLeg", activation=node_genome[n].fn); n+=1
        
    bodyID = 101 if c.use_obstacles else 1

    if (c.use_cpg and c.num_sensor_neurons > 5) or ( not c.use_cpg and c.num_sensor_neurons > 4):
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "BackLegRot_BackLeg", bodyID=bodyID, activation=node_genome[n].fn); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "FrontLegRot_FrontLeg", bodyID=bodyID, activation=node_genome[n].fn); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "LeftLegRot_LeftLeg", bodyID=bodyID, activation=node_genome[n].fn); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "LeftLegRot_LeftLeg", bodyID=bodyID, activation=node_genome[n].fn); n+=1

    if (c.use_cpg and c.num_sensor_neurons > 9) or ( not c.use_cpg and c.num_sensor_neurons > 8):
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "Torso_BackLegRot", bodyID=bodyID, activation=node_genome[n].fn); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "Torso_FrontLegRot", bodyID=bodyID, activation=node_genome[n].fn); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "Torso_LeftLegRot", bodyID=bodyID, activation=node_genome[n].fn); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "Torso_RightLegRot", bodyID=bodyID, activation=node_genome[n].fn); n+=1

    if (c.use_cpg and c.num_sensor_neurons > 13) or ( not c.use_cpg and c.num_sensor_neurons > 12):
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "BackLeg_BackLowerLeg", bodyID=bodyID, activation=node_genome[n].fn); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "FrontLeg_FrontLowerLeg", bodyID=bodyID, activation=node_genome[n].fn); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "LeftLeg_LeftLowerLeg", bodyID=bodyID, activation=node_genome[n].fn); n+=1
        pyrosim.Send_Rotation_Sensor_Neuron(name = n , jointName = "RightLeg_RightLowerLeg", bodyID=bodyID, activation=node_genome[n].fn); n+=1
    if (c.use_cpg and c.num_sensor_neurons > 17) or ( not c.use_cpg and c.num_sensor_neurons > 16):
        pyrosim.Send_Base_Velocity_Sensor_Neuron(name = n , bodyID=bodyID, activation=node_genome[n].fn); n+=1


    if c.use_cpg:
        pyrosim.Send_CPG(name = n, activation=node_genome[n].fn ); n+=1

    # -Hidden
    for neuron in node_genome:
        if neuron.type == 2: # Hidden
            pyrosim.Send_Hidden_Neuron(name = neuron.id, activation=neuron.fn)
        
    # -Output
    pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_BackLegRot", activation=node_genome[n].fn); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_FrontLegRot", activation=node_genome[n].fn); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_LeftLegRot", activation=node_genome[n].fn); n+=1
    pyrosim.Send_Motor_Neuron( name = n , jointName = "Torso_RightLegRot", activation=node_genome[n].fn); n+=1
    
    if (c.use_cpg and c.num_motor_neurons > 5) or (not c.use_cpg and c.num_motor_neurons > 4):
        pyrosim.Send_Motor_Neuron( name = n , jointName = "BackLeg_BackLowerLeg", activation=node_genome[n].fn); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "FrontLeg_FrontLowerLeg", activation=node_genome[n].fn); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "LeftLeg_LeftLowerLeg", activation=node_genome[n].fn); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "RightLeg_RightLowerLeg", activation=node_genome[n].fn); n+=1
    if (c.use_cpg and c.num_motor_neurons > 9) or (not c.use_cpg and c.num_motor_neurons > 8):
        pyrosim.Send_Motor_Neuron( name = n , jointName = "BackLegRot_BackLeg", activation=node_genome[n].fn); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "FrontLegRot_FrontLeg", activation=node_genome[n].fn); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "LeftLegRot_LeftLeg", activation=node_genome[n].fn); n+=1
        pyrosim.Send_Motor_Neuron( name = n , jointName = "RightLegRot_RightLeg", activation=node_genome[n].fn); n+=1


    # Synapses:
    # fully connected:
    for synapse in connection_genome:
            if synapse.enabled:
                pyrosim.Send_Synapse(sourceNeuronName = synapse.fromNode.id, targetNeuronName = synapse.toNode.id, weight = synapse.weight)

    pyrosim.End()
    
    while not os.path.exists(f"brain{id}.nndf"):
        time.sleep(0.01)
        
    if False:
        num = len([n for n in os.listdir('tmp') if os.path.isfile(n)])
        os.system(f"copy brain{id}.nndf tmp\\{id}.nndf")
        visualize_network( sample=True, sample_point=[0.1, -0.1, .25, -.25], use_radial_distance=False, save_name=f"tmp/{id}_{num}.png", show_weights=False)
