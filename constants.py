
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def identity(x):
    return x

gravity = 0, 0, -9.8

simulation_fps = 240
simulation_length = 2000

motor_max_force = 50.
motor_joint_range = 0.25

num_gens = 10
pop_size = 4

num_motor_neurons = 8
num_sensor_neurons = 4

hidden_nodes_at_start = 0

show_debug = False


do_crossover = True
use_speciation = True
use_map_elites = False
allow_recurrent = False
species_selection_ratio=.5
species_target = 6
species_threshold_delta = .25
init_species_threshold = 2.5
max_weight = 3.0
weight_threshold = 0
weight_mutation_max = 2
prob_random_restart =.001
prob_weight_reinit = 0.0
prob_reenable_connection = 0.1
species_stagnation_threshold = 20
fitness_threshold = 1
init_connection_probability = 1
within_species_elitism = 1
population_elitism = 1
activations = [tanh]
novelty_selection_ratio_within_species = 0.0
novelty_adjusted_fitness_proportion = 0.0
novelty_k = 5
novelty_archive_len = 5
curriculum = []
auto_curriculum = 0
num_workers = 4 

prob_mutate_activation = .1 
prob_mutate_weight = .35
prob_add_connection = .1 
prob_add_node = .1
prob_remove_node = 0.0
prob_disable_connection = .3

use_dynamic_mutation_rates = False
dynamic_mutation_rate_end_modifier = .1
use_multithreading = False
output_activation = tanh
save_progress_images = False

hidden_nodes_at_start = 0
allow_input_activation_mutation = False

use_input_bias = False
num_sensor_neurons = 4 # x,y,bias,d
num_motor_neurons = 4 

# Autoencoder novelty
autoencoder_frequency = -1

# clustering coefficent 
clustering_fitness_ratio = 0