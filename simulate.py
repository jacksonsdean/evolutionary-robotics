import sys
from simulation import Simulation
from generate import Generate
if __name__ == '__main__':
    args = sys.argv[1:]
    solution_id = 0
    if "--generate" in args:
        Generate()
    if "--id" in args:
        i = args.index("--id")
        solution_id = args[i+1]
    headless_mode = "DIRECT" in args
    sim = Simulation(headless_mode, solution_id)
    print("Running simulation with id:", solution_id)
    sim.run()
    sim.get_fitness()