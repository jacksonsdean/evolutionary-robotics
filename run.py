#%%
from search import main as search

class Arguments:
    pass

# %%
args = Arguments()
args.generations = 200
args.pop = 30
args.species = 3
args.experiment_runs = 5
args.alg = "hyperneat"
args.generate=False
args.debug = False
args.do_bootstrap = True
args.experiment_file = "experiments/num_phen_nodes.json"
search(args)
