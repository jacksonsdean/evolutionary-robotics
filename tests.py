#%%
# import constants as c
from neat import NEAT, mutate
import simulate

from  neat_genome import *

from util import visualize_network
#%%
neat = NEAT()

c.num_sensor_neurons = 2
c.num_motor_neurons = 2
c.init_connection_probability = 1.0
Node.current_id = c.num_sensor_neurons + c.num_motor_neurons
genome1 = Genome()

visualize_network(genome1)
genome1.add_node()
visualize_network(genome1)
genome1.add_node()
visualize_network(genome1)
genome1.add_node()
visualize_network(genome1)
# mutate(genome1, Genome, neat.get_mutation_rates())
# mutate(genome1, Genome, neat.get_mutation_rates())

# genome2 = Genome()
# child = crossover(genome1, genome2)
# visualize_network(genome1, sample=True)
# visualize_network(child, sample=True)
# %%
