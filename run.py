#%%
from search import main as search

class Arguments:
    pass

# %%
args = Arguments()
args.generations = 2
args.pop = 10
args.species = 2
args.experiment_runs = 2
args.alg = ""
args.generate = False
args.debug = False
args.print = False
args.do_bootstrap = False
args.experiment_file = "experiments/algorithm.json"
search(args)
