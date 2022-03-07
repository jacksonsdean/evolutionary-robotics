from parallel_hillclimber import ParallelHillClimber
import constants as c

phc = ParallelHillClimber(c.num_gens, c.pop_size)
phc.evolve()
phc.show_best()
