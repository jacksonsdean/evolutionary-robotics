import sys
from simulation import Simulation
from generate import Generate
if __name__ == '__main__':
    args = sys.argv[1:]
    solution_id = 0
    save_as_best = False
    if "--generate" in args:
        Generate()
    if "--id" in args:
        i = args.index("--id")
        solution_id = args[i+1]
    if "--best" in args:
        save_as_best = True
    headless_mode = "DIRECT" in args
    sim = Simulation(headless_mode, solution_id, save_as_best)
    print("Running simulation with id:", solution_id)
    sim.run()
    sim.get_fitness()