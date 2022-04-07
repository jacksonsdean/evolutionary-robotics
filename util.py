import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
# import pygraphviz as pgv 
# from node_functions import *
# from networkx.drawing.nx_agraph import graphviz_layout
import sys
import inspect
import random
# from evolution_util import *
import zlib
# import sbi

import constants as c
    
def choose_random_function():
    return random.choice(c.activations)


def name_to_fn(name):
    fns = inspect.getmembers(sys.modules["node_functions"])
    fns.extend([("", None)])
    def avg_pixel_distance_fitness():
            pass
    fns.extend([("avg_pixel_distance_fitness", avg_pixel_distance_fitness)])
    return fns[[f[0] for f in fns].index(name)][1]
    
def visualize_network(individual,sample_point=[.25]*c.num_sensor_neurons, color_mode="L", visualize_disabled=False, layout='multi', sample=False, show_weights=False, use_inp_bias=False, use_radial_distance=True, save_name=None, extra_text=None):
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
    use_curved = show_weights or individual.count_layers()<3

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
        plt.close()
    else:
        plt.show()
        plt.close()


    ""
    # labels = nx.get_edge_attributes(G,'weight')



def plot_mean_and_bootstrapped_ci_over_time(input_data = None, dataset=None, name = "change me", x_label = "change me", y_label="change me", y_limit = None, plot_bootstrap = True):
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
    input_data = np.array(input_data)
    if isinstance(name, str): name = [name]; input_data = [input_data]

    # for this_input_data, this_name in zip(input_data, name):
    for index, this_name in enumerate(name):
        print("plotting",this_name)
        this_input_data = dataset[index]
        total_generations = this_input_data.shape[1]
        if(plot_bootstrap):
            boostrap_ci_generation_found = np.zeros((2,total_generations))
            for this_gen in range(total_generations):
                boostrap_ci_generation_found[:,this_gen] = bootstrap.ci(this_input_data[:,this_gen], np.mean, alpha=0.05)


        ax.plot(np.arange(total_generations), np.mean(this_input_data,axis=0), label = this_name) # plot the fitness over time
        if plot_bootstrap:
            ax.fill_between(np.arange(total_generations), boostrap_ci_generation_found[0,:], boostrap_ci_generation_found[1,:],alpha=0.3) # plot, and fill, the confidence interval for fitness over time
        ax.set_xlabel(x_label) # add axes labels
        ax.set_ylabel(y_label)
        if y_limit: ax.set_ylim(y_limit[0],y_limit[1])
        plt.legend(loc='best'); # add legend


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
    