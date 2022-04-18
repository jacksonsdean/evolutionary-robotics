import random
from neat_solution import NeatSolution
import copy
import os
import platform

class ParallelHillClimberWithAugmentingTopologies():
    def __init__(self, num_gens=10, pop_size=2, epsilon=.01, debug_output=False) -> None:
        self.parents = {}
        self.pop_size = pop_size
        self.num_gens = num_gens
        self.gen = 0
        self.next_available_id = 0
        self.epsilon = epsilon
        self.debug_output = debug_output
        
        for index in range(pop_size):
            self.parents[index] = NeatSolution(self.next_available_id)
            self.next_available_id += 1
            
        # remove temp files:
        if platform.system() == "Windows":
            os.system("del brain*.nndf")
            os.system("del body*.urdf")
            os.system("del fitness*.txt")
        else:
            os.system("rm brain*.nndf")
            os.system("rm body*.urdf")
            os.system("rm fitness*.txt")
        

    def evolve(self):
        self.evaluate(self.parents)
        for self.gen in range(self.num_gens):
            self.run_one_generation()

    def run_one_generation(self):
        self.spawn()
        self.mutate()
        self.evaluate(self.children)
        self.print_fitnesses()
        self.select()
    
    def get_best(self):
        lowest = min(self.parents.keys(), key=(lambda k: self.parents[k].fitness))
        return lowest
    
    def print_best(self):
        lowest = self.get_best()
        print("Best:", lowest, self.parents[lowest].fitness)
        
    def show_best(self):
        print()
        self.print_best()
        self.parents[self.get_best()].start_simulation(False, self.debug_output, True)

    def print_fitnesses(self):
        print("Generation:", self.gen)
        for key in self.parents.keys():
            print(f"parent {key} fitness:", self.parents[key].fitness, end="\t\t|\t")
            print(f"child {key} fitness:", self.children[key].fitness)
        print(f"Best in gen {self.gen}: {self.get_best()} ({self.parents[self.get_best()].fitness})")
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
            solutions[key].start_simulation(True, self.debug_output)
        for key in solutions.keys():
            solutions[key].wait_for_simulation()
            
    def select(self):
        for key in self.parents.keys():
            if not(self.children[key].fitness <= self.get_best()) and random.random() < self.epsilon:
                # Backwards step
                if self.children[key].fitness > self.parents[key].fitness:
                    self.parents[key] = self.children[key] 
            else:
                if self.children[key].fitness < self.parents[key].fitness:
                    self.parents[key] = self.children[key] 