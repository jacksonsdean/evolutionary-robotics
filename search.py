from parallel_hillclimber_at import ParallelHillClimberWithAugmentingTopologies
import constants as c

phc = ParallelHillClimberWithAugmentingTopologies(c.num_gens, c.pop_size, 0.1, c.show_debug)
phc.evolve()
phc.show_best()
