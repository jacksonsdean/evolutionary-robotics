from activations import *
import constants as c

def apply():
    c.init_species_threshold = 5.
    c.species_threshold_delta = .5
    c.activations = [sin, gaussian, sigmoid, abs]
    c.init_connection_probability = .75
    c.hidden_nodes_at_start = 2
    c.weight_threshold = 0.2
    c.allow_recurrent=False
    
    c.num_hn_hidden_layers = 2 # 2 works well
    c.num_hn_hidden_nodes_per_layer = [4, 4]
    c.num_hn_inputs = 4 # (x1, y1, x2, y2)
    c.num_hn_outputs = 1
    # c.substrate_type = "grid"
    c.substrate_type = "sandwich"

    c.hidden_nodes_at_start = 0
    
    c.max_phen_weight = c.max_weight
    c.max_gen_weight = 3
    # c.num_sensor_neurons = 4
    # c.num_motor_neurons = 4