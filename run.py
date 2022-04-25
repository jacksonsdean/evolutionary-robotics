#%%
from search import main as search

class Arguments:
    pass

# %%
args = Arguments()
args.generations = 100
args.pop = 20
args.species = 4
args.experiment_runs = 3
args.alg = "hyperneat"
args.generate = False
args.debug = False
args.do_bootstrap = False
args.experiment_file = "experiments/neat_vs_hyperneat.json"
search(args)
