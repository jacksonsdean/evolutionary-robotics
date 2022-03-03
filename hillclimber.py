from solution import Solution
import copy

class HillClimber():
    def __init__(self, num_gens=10) -> None:
        self.parent = Solution()
        self.num_gens = num_gens

    def evolve(self):
        self.parent.evaluate(False)
        for gen in range(self.num_gens):
            self.run_one_generation()
            
    def run_one_generation(self):
        self.spawn()
        self.mutate()
        self.child.evaluate()
        self.print_fitnesses()
        self.select()
    
    def show_best(self):
        self.parent.evaluate(False)

    def print_fitnesses(self):
        print("\n"+"-"*60+"\nparent fitness:", self.parent.fitness, end=" | ")
        print("child fitness:", self.child.fitness, end="\n"+"-"*60+"\n\n")

    def spawn(self):
        self.child = copy.deepcopy(self.parent)

    def mutate(self):
        self.child.mutate()

    def select(self):
        if self.child.fitness < self.parent.fitness:
            self.parent = self.child 