#%%
from search import main as search

class Arguments:
    pass

# %%
args = Arguments()
args.generations = 2
args.pop = 10
args.species = 2
args.experiment_runs = 1
args.alg = "neat"
args.generate = False
args.debug = False
args.print = True
args.do_bootstrap = False
args.experiment_file = "experiments/neat_vs_hyperneat.json"
search(args)
