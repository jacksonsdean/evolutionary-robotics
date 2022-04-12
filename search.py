import os
from tkinter.tix import Tree
from hyperneat import HyperNEAT
from neat import NEAT
import constants as c
import hyperneat_constants as hc
import argparse

# parse args
parser = argparse.ArgumentParser(description='Run search on the robot controller.')
parser.add_argument('-d','--debug', action='store_true', help='Show debug messages.')
parser.add_argument('-g','--generate', action='store_true', help='Generate new world first.')
parser.add_argument('-t','--generations', action='store', help='Number of generations to run.')
parser.add_argument('-p','--pop', action='store', help='Population size.')
parser.add_argument('-s','--species', action='store', help='Number of species.')
parser.add_argument('-a','--alg', action='store', help='Algorithm to use.')

# parser.add_argument('-o','--obstacles', action='store_true', help='Use obstacles.')

args = parser.parse_args()


if args.generations:
    c.num_gens = int(args.generations)

if args.pop:
    c.pop_size = int(args.pop)
if args.species:
    c.species_target = int(args.species)


if args.generate:
    os.system("python generate.py")


    
if args.alg == "neat":
    neat = NEAT(args.debug)
elif args.alg == "hyperneat":
    hc.apply()
    neat = HyperNEAT(args.debug)

try:
    neat.evolve()
except KeyboardInterrupt:
    print("Stopping early...")
    neat.save_best_network_image(True)
    
neat.show_best()
neat.show_fitness_curve()
neat.show_diversity_curve()
