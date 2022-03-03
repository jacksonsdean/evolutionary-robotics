from hillclimber import HillClimber
import constants as c

hc = HillClimber(c.num_gens)
hc.evolve()
hc.show_best()
