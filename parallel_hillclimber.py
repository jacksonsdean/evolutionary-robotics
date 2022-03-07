from solution import Solution
import copy
import os
import platform

class ParallelHillClimber():
    def __init__(self, num_gens=10, pop_size=2) -> None:
        self.parents = {}
        self.pop_size = pop_size
        self.num_gens = num_gens
        self.next_available_id = 0
        
        for index in range(pop_size):
            self.parents[index] = Solution(self.next_available_id)
            self.next_available_id += 1
            
        # remove temp files:
        if platform.system() == "Windows":
            os.system("del brain*.nndf")
            os.system("del fitness*.txt")
        else:
            os.system("rm brain*.nndf")
            os.system("rm fitness*.txt")
        

    def evolve(self):
        self.evaluate(self.parents)
        for gen in range(self.num_gens):
            self.run_one_generation()

    def run_one_generation(self):
        self.spawn()
        self.mutate()
        self.evaluate(self.children)
        self.print_fitnesses()
        self.select()
    
    def show_best(self):
        lowest = min(self.parents.keys(), key=(lambda k: self.parents[k].fitness))
        print("Best:", lowest, self.parents[lowest].fitness)
        self.parents[lowest].start_simulation(False)

    def print_fitnesses(self):
        print()
        for key in self.parents.keys():
            print("-"*60+f"\nparent {key} fitness:", self.parents[key].fitness, end=" | ")
            print(f"child {key} fitness:", self.children[key].fitness)
        print()

    def spawn(self):
        self.children = {}
        for key in self.parents.keys():
            self.children[key] = copy.deepcopy(self.parents[key])
            self.children[key].set_id(self.next_available_id)
            self.next_available_id += 1
            
    def mutate(self):
        for key in self.children.keys():
            self.children[key].mutate()

    def evaluate(self, solutions):
        for key in solutions.keys():
            solutions[key].start_simulation(True)
        for key in solutions.keys():
            solutions[key].wait_for_simulation()

    def select(self):
        for key in self.parents.keys():
            if self.children[key].fitness < self.parents[key].fitness:
                self.parents[key] = self.children[key] 