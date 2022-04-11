import math
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange
from neat_genome import Connection, Genome, crossover,Node
from util import choose_random_function, get_avg_number_of_connections, get_avg_number_of_hidden_nodes, get_max_number_of_connections, get_max_number_of_hidden_nodes, visualize_network
from species import *
import copy 
from neat import NEAT, classic_selection_and_reproduction, update_solution_archive
import random
import copy
import os
import platform
import constants as c

class HyperNEAT(NEAT):
    ...