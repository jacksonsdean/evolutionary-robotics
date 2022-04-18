#%%
from search import main as search

class Arguments:
    pass

# %%
args = Arguments()
args.generations = 100
args.pop = 40
args.species = 5
args.experiment_runs = 10
args.alg = "neat"
# args.alg = "hyperneat"
args.generate = False
args.debug = False
args.do_bootstrap = False
args.experiment_file = "experiments/max_weight.json"
search(args)
