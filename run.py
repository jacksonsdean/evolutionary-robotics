#%%
from search import main as search

class Arguments:
    ...
# %%
args = Arguments()
args.generations = 100
args.pop = 20
args.species = 2
args.experiment_runs = 3
args.alg = "neat"
args.generate=False
args.debug = False

search(args)
