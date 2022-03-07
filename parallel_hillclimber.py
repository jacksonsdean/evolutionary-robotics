from solution import Solution
import copy

class ParallelHillClimber():
    def __init__(self, num_gens=10, pop_size=2) -> None:
        self.parents = {}
        self.pop_size = pop_size
        self.num_gens = num_gens
        self.next_available_id = 0
        
        for index in range(pop_size):
            self.parents[index] = Solution(self.next_available_id)
            self.next_available_id += 1
        

    def evolve(self):
        # self.parent.evaluate(False)
        # for gen in range(self.num_gens):
            # self.run_one_generation()
        for index in range(self.pop_size):
            self.parents[index].start_simulation(True)
        
        for index in range(self.pop_size):
            self.parents[index].wait_for_simulation()

    def run_one_generation(self):
        self.spawn()
        self.mutate()
        self.child.evaluate()
        self.print_fitnesses()
        self.select()
    
    def show_best(self):
        # self.parent.evaluate(False)
        ...

    def print_fitnesses(self):
        print("\n"+"-"*60+"\nparent fitness:", self.parent.fitness, end=" | ")
        print("child fitness:", self.child.fitness, end="\n"+"-"*60+"\n\n")

    def spawn(self):
        self.child = copy.deepcopy(self.parent)
        self.child.set_id(self.next_available_id)
        self.next_available_id += 1

    def mutate(self):
        self.child.mutate()

    def select(self):
        if self.child.fitness < self.parent.fitness:
            self.parent = self.child 