import sys
from simulation import Simulation
from generate import Generate
import constants as c
if __name__ == '__main__':
    args = sys.argv[1:]
    solution_id = 0
    brain_path = None
    body_path = None
    save_video = None
    save_as_best = False

    if "--generate" in args:
        Generate()
        body_path="best_body.urdf"
        brain_path="best_brain.nndf"
        
    if "--id" in args:
        i = args.index("--id")
        solution_id = args[i+1]
    if "--best" in args:
        save_as_best = True
    if "--brain" in args:
        i = args.index("--brain")
        brain_path = args[i+1]
    if "--body" in args:
        i = args.index("--body")
        body_path = args[i+1]
        
    if "--length" in args:
        i = args.index("--length")
        c.simulation_length = int(args[i+1])

    if "--mp4" in args:
        i = args.index("--mp4")
        save_video = args[i+1]
        
    headless_mode = "DIRECT" in args
    sim = Simulation(headless_mode, solution_id, save_as_best, brain_path, body_path, save_video)
    print("Running simulation with id:", solution_id)
    sim.run()
    sim.get_fitness()