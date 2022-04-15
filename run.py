#%%
from search import main as search

class Arguments:
    ...
# %%
args = Arguments()
args.generations = 100
args.pop = 10
args.species = 2
args.experiment_runs = 4
args.alg = "neat"
args.generate=False
args.debug = False
args.do_bootstrap = True

search(args)
