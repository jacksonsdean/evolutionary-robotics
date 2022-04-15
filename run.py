#%%
from search import main as search

class Arguments:
    ...
# %%
args = Arguments()
args.generations = 2
args.pop = 4
args.species = 2
args.experiment_runs = 2
args.alg = "neat"
args.generate=False
args.debug = False
args.do_bootstrap = False

search(args)
