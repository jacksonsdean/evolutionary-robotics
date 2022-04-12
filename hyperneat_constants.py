from activations import *
import constants as c

def apply():
    c.init_species_threshold = 5.
    c.activations = [tanh, sin, gaussian, sigmoid, identity]
    c.init_connection_probability = 1.0
    
    c.weight_threshold = 0.0001

    c.num_hn_hidden_layers = 1
    c.num_hn_hidden_nodes_per_layer = 15
    c.num_hn_inputs = 4 # (x1, y1, x2, y2)
    c.num_hn_outputs = 1

            
    c.prob_mutate_activation = .5 
    c.prob_mutate_weight = .5
    c.prob_add_connection = .65
    c.prob_add_node = .8
    c.prob_remove_node = 0.55
    c.prob_disable_connection = .6

    c.hidden_nodes_at_start = 5