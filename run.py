#%%
from search import main as search

class Arguments:
    ...
# %%
args = Arguments()
args.generations = 100
args.pop = 5
args.species = 1
args.experiment_runs = 4
args.alg = "neat"
args.generate=False
args.debug = False
args.do_bootstrap = True
args.experiment_file = "experiments/weight_mutation_rate_low_mid.json"
search(args)
