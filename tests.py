#%%
# import constants as c
from matplotlib import pyplot as plt
from hyperneat import HyperNEAT, HyperNEATGenome
from neat import NEAT
from neat_genome import Genome as NEATGenome
import simulate

from  neat_genome import *

from util import visualize_hn_phenotype_network, visualize_network

# %%
# import hyperneat_constants as hc
# hc.apply()
# g = HyperNEATGenome()
# visualize_network(g)
# #%%
# vis = g.create_output_visualization(32, 32)
# plt.imshow(vis, vmin=-1, vmax=1)
# # plt.imshow(vis)
# plt.show()
# g.visualize_phenotype_network()   
#%%
# g.eval_substrate_simple()
# g.visualize_phenotype_network()   
# %%

# alg = HyperNEAT(False)
# parent1 = HyperNEATGenome()
# parent1.eval_substrate_simple()
# # parent1.visualize_phenotype_network()
# # parent1.visualize_genotype_network()
# parent2 = HyperNEATGenome()
# parent2.eval_substrate_simple()
# parent1.add_node()
# parent1.add_node()
# child = alg.crossover(parent1, parent2)
# child.eval_substrate_simple()
# # child.visualize_phenotype_network()
# child.visualize_genotype_network()
# plt.show()


#%% 
c.num_sensor_neurons = 3
c.num_motor_neurons = 3
c.init_connection_probability = 0
genome = NEATGenome()
print(genome.connection_genome)
visualize_network(genome, curved=True)
genome.add_connection()
visualize_network(genome, curved=True)
print(genome.connection_genome)
