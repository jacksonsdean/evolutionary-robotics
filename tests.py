#%%
# import constants as c
from neat import NEAT, mutate
import simulate

from  neat_genome import *

from util import visualize_network
#%%
neat = NEAT()

genome1 = Genome()
genome1.add_node()
genome1.add_node()
genome1.add_node()
mutate(genome1, Genome, neat.get_mutation_rates())
mutate(genome1, Genome, neat.get_mutation_rates())

genome2 = Genome()
child = crossover(genome1, genome2)
visualize_network(genome1, sample=True)
visualize_network(child, sample=True)
# %%
