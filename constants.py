from activations import *

gravity = 0, 0, -9.8

simulation_fps = 240
simulation_length = 2000

motor_max_force = 40.
motor_joint_range = .3

num_gens = 10
pop_size = 10

use_obstacles = False
max_obstacle_height = 0.10

# num_motor_neurons = 2  
# num_sensor_neurons = 2
num_motor_neurons = 12  
num_sensor_neurons = 16



num_hn_hidden_layers = 1
num_hn_hidden_nodes_per_layer = 10
num_hn_inputs = 4 # (x1, y1, x2, y2)
num_hn_outputs = 1

use_cpg = False
if use_cpg: num_sensor_neurons+=1
torso_weight = 3


hidden_nodes_at_start = 0
init_connection_probability = 1

species_target = 3
species_selection_ratio= .5
species_threshold_delta = .5
init_species_threshold = 7.5


do_crossover = True
use_speciation = True
use_map_elites = False
allow_recurrent = True

max_weight = 1.0
weight_threshold = 0
weight_mutation_max = .5
prob_random_restart =.001
prob_weight_reinit = 0.2
prob_reenable_connection = 0.1
species_stagnation_threshold = 20
fitness_threshold = 1e10
within_species_elitism = 1
population_elitism = 1
activations = [tanh]
# activations = [tanh, sin, sigmoid, identity]
novelty_selection_ratio_within_species = 0.0
novelty_adjusted_fitness_proportion = 0.0
novelty_k = 5
novelty_archive_len = 5
curriculum = []
auto_curriculum = 0
num_workers = 4 

prob_mutate_activation = .1 
prob_mutate_weight = .2
prob_add_connection = .35
prob_add_node = .4
prob_remove_node = 0.35
prob_disable_connection = .3

use_dynamic_mutation_rates = False
dynamic_mutation_rate_end_modifier = .1
use_multithreading = False
output_activation = tanh
save_progress_images = False

allow_input_activation_mutation = False

use_input_bias = False

# Autoencoder novelty
autoencoder_frequency = -1

# clustering coefficent 
clustering_fitness_ratio = 0

# from activations import *

# gravity = 0, 0, -9.8

# simulation_fps = 240
# simulation_length = 2000

# motor_max_force = 70.
# motor_joint_range = .35

# num_gens = 10
# pop_size = 10

# torso_weight = 3.0

# use_obstacles = False
# max_obstacle_height = 0.10

# use_cpg = True
# num_motor_neurons = 12  
# num_sensor_neurons = 8 
# if use_cpg:num_sensor_neurons+=1


# hidden_nodes_at_start = 0
# init_connection_probability = .35ZZZ

# species_target = 3
# species_selection_ratio= .5
# species_threshold_delta = .1
# init_species_threshold = .75
# species_stagnation_threshold = 20


# do_crossover = True
# use_speciation = True
# use_map_elites = False
# allow_recurrent = True
# fitness_threshold = 1e10
# max_weight = 1.0
# weight_threshold = 0.0001
# weight_mutation_max = .5
# prob_mutate_weight = .2
# prob_weight_reinit = .3
# prob_random_restart =.001
# # prob_weight_reinit = 0.01
# prob_reenable_connection = 0.1
# prob_mutate_activation = .1 
# prob_add_connection = .45
# prob_add_node = .8
# prob_remove_node = 0.45
# prob_disable_connection = .4


# activations = [tanh]
# # activations = [tanh, sin, cos, identity, sigmoid, step, gaussian]
# within_species_elitism = 0
# population_elitism = 1
# novelty_selection_ratio_within_species = 0.0
# novelty_adjusted_fitness_proportion = 0.0
# novelty_k = 5
# novelty_archive_len = 5
# num_workers = 4 

# use_dynamic_mutation_rates = False
# dynamic_mutation_rate_end_modifier = .1
# use_multithreading = False
# output_activation = tanh
# save_progress_images = False

# allow_input_activation_mutation = False

# use_input_bias = False

# # Autoencoder novelty
# autoencoder_frequency = -1

# # clustering coefficent 
# clustering_fitness_ratio = 0