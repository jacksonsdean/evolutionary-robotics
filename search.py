from neat import NEAT
import constants as c

phc = NEAT(c.show_debug)
phc.evolve()
phc.show_best()
