from solution import Solution

class HillClimber():
    def __init__(self, num_gens=10) -> None:
        self.parent = Solution()
        self.num_gens = num_gens

    def evolve(self):
        for gen in range(self.num_gens):
            self.run_one_generation()
            
    def run_one_generation(self):
        self.spawn()
        self.mutate()
        self.child.evaluate()
        self.select()
    
    def spawn(self):
        ...
    def mutate(self):
        ...
    def select(self):
        ...