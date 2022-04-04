import os
from neat import NEAT
import constants as c

import argparse

# parse args
parser = argparse.ArgumentParser(description='Run search on the robot controller.')
parser.add_argument('-d','--debug', action='store_true', help='Show debug messages.')
parser.add_argument('-g','--generate', action='store_true', help='Generate new world first.')
parser.add_argument('-l','--generations', action='store', help='Number of generations to run.')
parser.add_argument('-p','--pop', action='store', help='Population size.')
args = parser.parse_args()


if args.generations:
    c.num_gens = int(args.generations)

if args.pop:
    c.pop_size = int(args.pop)

if args.generate:
    os.system("python generate.py")


phc = NEAT(args.debug)
phc.evolve()
phc.show_best()
