#%%
# import constants as c
from matplotlib import pyplot as plt
from hyperneat import HyperNEATGenome
from neat import NEAT
import simulate

from  neat_genome import *

from util import visualize_network

# %%
import hyperneat_constants as hc
hc.apply()
g = HyperNEATGenome()
visualize_network(g)
#%%
# vis = g.create_output_visualization(32, 32)
plt.imshow(vis, vmin=-1, vmax=1)
# plt.imshow(vis)
plt.show()
g.visualize_phenotype_network()   
#%%
g.eval_substrate_simple()
g.visualize_phenotype_network()   
# %%
