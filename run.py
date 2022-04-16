#%%
from search import main as search

class Arguments:
    pass

# %%
args = Arguments()
args.generations = 10
args.pop = 4
args.species = 1
args.experiment_runs = 5
args.alg = "hyperneat"
args.generate=False
args.debug = False
args.do_bootstrap = True
args.experiment_file = "experiments/control.json"
search(args)
