#%%
from search import main as search

class Arguments:
    pass

# %%
args = Arguments()
args.generations = 100
args.pop = 20
args.species = 3
args.experiment_runs = 4
# args.alg = "neat"
args.alg = "hyperneat"
args.generate = False
args.debug = False
args.do_bootstrap = False
args.experiment_file = "experiments/num_phen_nodes.json"
search(args)
