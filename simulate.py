import sys
from simulation import Simulation
from generate import Generate
import constants as c
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
    if "--brain" in args:
        i = args.index("--brain")
        brain_path = args[i+1]
    else:
        brain_path = None
    if "--body" in args:
        i = args.index("--body")
        body_path = args[i+1]
        
    if "--length" in args:
        i = args.index("--length")
        c.simulation_length = int(args[i+1])
        
    else:
        body_path = None
    headless_mode = "DIRECT" in args
    sim = Simulation(headless_mode, solution_id, save_as_best, brain_path, body_path)
    print("Running simulation with id:", solution_id)
    sim.run()
    sim.get_fitness()