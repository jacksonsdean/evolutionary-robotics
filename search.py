from parallel_hillclimber import ParallelHillClimber
import constants as c

phc = ParallelHillClimber(c.num_gens, c.pop_size, 0.1, c.show_debug)
phc.evolve()
phc.show_best()
