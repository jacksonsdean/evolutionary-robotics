from neat import NEAT
import constants as c


phc = NEAT(True)
phc.evolve()
phc.show_best()
