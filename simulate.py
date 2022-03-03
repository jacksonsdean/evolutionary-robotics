import sys
from simulation import Simulation
from generate import Generate
if __name__ == '__main__':
    args = sys.argv[1:]
    if "--generate" in args:
        Generate()
    sim = Simulation()
    sim.run()
    sim.get_fitness()