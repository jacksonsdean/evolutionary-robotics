import math
import os
import time
from matplotlib import pyplot as plt
import numpy as np
from tqdm import trange
from neat_genome import Connection, Genome, crossover
from util import choose_random_function, get_avg_number_of_connections, get_avg_number_of_hidden_nodes, get_max_number_of_connections, get_max_number_of_hidden_nodes, visualize_network
from species import *
import copy 

import random
import copy
import os
import platform
import constants as c

class NEAT():
    def __init__(self, debug_output=False) -> None:
        self.gen = 0
        self.next_available_id = 0
        self.debug_output = debug_output
        self.all_species = []
        
        self.diversity_over_time = np.zeros(c.num_gens,dtype=float)
        self.species_over_time = np.zeros(c.num_gens,dtype=np.float)
        self.species_threshold_over_time = np.zeros(c.num_gens, dtype=np.float)
        self.nodes_over_time = np.zeros(c.num_gens, dtype=np.float)
        self.connections_over_time = np.zeros(c.num_gens, dtype=np.float)
        self.fitness_over_time = np.zeros(c.num_gens, dtype=np.float)
        self.solutions_over_time = []
        self.species_champs_over_time = []
            
        self.solution_generation = -1
        self.species_threshold = c.init_species_threshold
        self.population = []
        self.solution = None
        
        self.solution_fitness = 0
        # remove temp files:
        if platform.system() == "Windows":
            os.system("del brain*.nndf")
            os.system("del body*.urdf")
            os.system("del fitness*.txt")
            os.system("del tmp*.txt")
        else:
            os.system("rm brain*.nndf")
            os.system("rm body*.urdf")
            os.system("rm fitness*.txt")
            os.system("rm tmp*.txt")
    
    
    def get_mutation_rates(self):
        run_progress = self.gen / c.num_gens
        if(c.use_dynamic_mutation_rates):
            end_mod = c.dynamic_mutation_rate_end_modifier
            prob_mutate_activation   = c.prob_mutate_activation   - (c.prob_mutate_activation    - end_mod * c.prob_mutate_activation)   * run_progress
            prob_mutate_weight       = c.prob_mutate_weight       - (c.prob_mutate_weight        - end_mod * c.prob_mutate_weight)       * run_progress
            prob_add_connection      = c.prob_add_connection      - (c.prob_add_connection       - end_mod * c.prob_add_connection)      * run_progress
            prob_add_node            = c.prob_add_node            - (c.prob_add_node             - end_mod * c.prob_add_node)            * run_progress
            prob_remove_node         = c.prob_remove_node         - (c.prob_remove_node          - end_mod * c.prob_remove_node)         * run_progress
            prob_disable_connection  = c.prob_disable_connection  - (c.prob_disable_connection   - end_mod * c.prob_disable_connection)  * run_progress
            weight_mutation_max      = c.weight_mutation_max      - (c.weight_mutation_max       - end_mod * c.weight_mutation_max)      * run_progress
            prob_reenable_connection = c.prob_reenable_connection - (c.prob_reenable_connection  - end_mod * c.prob_reenable_connection) * run_progress
            return  prob_mutate_activation, prob_mutate_weight, prob_add_connection, prob_add_node, prob_remove_node, prob_disable_connection, weight_mutation_max, prob_reenable_connection
        else:
            return  c.prob_mutate_activation, c.prob_mutate_weight, c.prob_add_connection, c.prob_add_node, c.prob_remove_node, c.prob_disable_connection, c.weight_mutation_max, c.prob_reenable_connection

    def update_fitnesses_and_novelty(self):
        for ind in self.population:
            ind.start_simulation(True, self.debug_output)
        for ind in self.population:
            ind.wait_for_simulation()

        for i, g in enumerate(self.population):
            self.population[i].update_with_fitness(g.fitness, count_members_of_species(self.population, self.population[i].species_id))
        
        if(c.novelty_selection_ratio_within_species > 0 or c.novelty_adjusted_fitness_proportion > 0):
            # novelties = novelty_ae.get_ae_novelties(self.population)
            novelties = np.zeros(len(self.population))
            for i, n in enumerate(novelties):
                self.population[i].novelty = n
                self.novelty_archive = update_solution_archive(self.novelty_archive, self.population[i], c.novelty_archive_len, c.novelty_k)
        
        for i in range(len(self.population)):
            if c.novelty_adjusted_fitness_proportion > 0:
                global_novelty = np.mean([g.novelty for g in self.novelty_archive])
                if(global_novelty==0): global_novelty=0.001
                adj_fit = self.population[i].adjusted_fitness
                adj_novelty =  self.population[i].novelty / global_novelty
                prop = c.novelty_adjusted_fitness_proportion
                self.population[i].adjusted_fitness = (1-prop) * adj_fit  + prop * adj_novelty 
                

    def print_fitnesses(self):
        print("Generation:", self.gen)
        for individual in self.population:
            print(f"Individual {individual.id} (species: {individual.species_id}) fitness:",individual.fitness)
        print(f"Best in gen {self.gen}: {self.get_best().id} ({self.get_best().fitness})")
        print(f"Average fitness: {np.mean([i.fitness for i in self.population])}")
        print(f"Average adjusted fitness: {np.mean([i.adjusted_fitness for i in self.population])}")
        num_species = count_number_of_species(self.population)
        print(f"Number of species: {num_species} | threshold: {self.species_threshold}")
        print(f"Best species (avg. fitness): {sorted(self.all_species, key=lambda x: x.avg_fitness if x.population_count > 0 else 0, reverse=True)[0].id}")
        print(f"Diversity (std, mean, max): {calculate_diversity_full(self.population, self.all_species)}")
        print(f"Avg. Connections: {get_avg_number_of_connections(self.population)} | Max Connections: {get_max_number_of_connections(self.population)}")
        print(f"Avg. Hidden Nodes: {get_avg_number_of_hidden_nodes(self.population)} | Max Nodes: {get_max_number_of_hidden_nodes(self.population)}")
        print()


    def neat_selection_and_reproduction(self):
        new_children = []
        # for i in self.population:
            # print("Individual:", i.id, "adjusted fitness:", i.adjusted_fitness)
        global_average_fitness = np.mean([i.adjusted_fitness for i in self.population])
        for sp in self.all_species:
            sp.population_count = count_members_of_species(self.population, sp.id) 
            if(sp.population_count<=0): sp.allowed_offspring = 0; continue
            members = get_members_of_species(self.population, sp.id)
            sp.update(global_average_fitness, members, self.gen, c.species_stagnation_threshold, c.pop_size)

        normalize_species_offspring(self.all_species, c)
        for sp in self.all_species:
            if(sp.population_count<=0): continue
            members = get_members_of_species(self.population, sp.id) # maintains sort order
            sp.population_count = len(members)
            sp.current_champ = members[0] # still sorted from before
            if(c.within_species_elitism > 0  and sp.allowed_offspring > 0):
                n_elites = min(sp.population_count, c.within_species_elitism, sp.allowed_offspring) 
                for i in range(n_elites):
                    # Elitism: add the elite and make one less offspring
                    # new_children.append(copy.deepcopy(members[i]))
                    new_children.append(copy.copy(members[i]))
                    sp.allowed_offspring-=1
            if(len(members)>1):
                new_members = []
                fitness_selected = round((1-c.novelty_selection_ratio_within_species) * c.species_selection_ratio * len(members)) 
                new_members = members[:fitness_selected] # truncation selection
                if (c.novelty_selection_ratio_within_species >0):
                    novelty_members = sorted(members, key=lambda x: x.novelty, reverse=True) 
                    novelty_selected = round(c.novelty_selection_ratio_within_species * c.species_selection_ratio * len(members)) 
                    new_members.extend(novelty_members[:novelty_selected+1]) # truncation selection
                
                members = new_members
                # members = tournament_selection(members, c, True, override_no_elitism=True) # tournament selection
            
            for i in range(sp.allowed_offspring):
                # inheritance
                parent1 = np.random.choice(members, size=1)[0] # pick 1 random parent
                #crossover
                if(c.do_crossover):
                    if(np.random.rand()<.001): # cross-species crossover (.001 in s/m07)
                        other_id = -1
                        for sp2 in self.all_species:
                            if count_members_of_species(self.population, sp2.id) > 0 and sp2.id!=sp.id:
                                other_id = sp2.id
                        if(other_id>-1): members = get_members_of_species(self.population, other_id)
                    parent2 = np.random.choice(members, size=1)[0] # allow parents to crossover with themselves
                    child = crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)    

                mutate(child, Genome, self.get_mutation_rates())
                new_children.extend([child]) # add children to the new_children list
                
        return new_children

    def evolve(self):
        for i in range(c.pop_size): # only create parents for initialization (the mu in mu+lambda)
            self.population.append(Genome()) # generate new random individuals as parents
            
        if c.use_speciation:
            assign_species(self.all_species, self.population, self.species_threshold, Species)

        # Run NEAT
        for self.gen in range(c.num_gens):
            self.run_one_generation()
            

    def run_one_generation(self):
        # update all ids:
        for ind in self.population:
            ind.set_id(Genome.get_id())
        #------------#
        # assessment # //TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        #------------#
        self.update_fitnesses_and_novelty()
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True) # sort by fitness
        self.solution = self.population[0]
        self.print_fitnesses()

        # if self.gen == 0:
            # self.show_best()
            
        # dynamic mutation rates
        mutation_rates = self.get_mutation_rates()

        # the modification procedure
        new_children = [] # keep children separate

        for i in range(c.population_elitism):
            new_children.append(copy.deepcopy(self.population[i])) # keep most fit individuals without mutating (should already be sorted)

        if(c.use_speciation):
            new_children.extend(self.neat_selection_and_reproduction())
            assign_species(self.all_species, new_children, self.species_threshold, Species) # assign new species ids
        else:
            new_children.extend(classic_selection_and_reproduction(c, self.population, self.all_species, self.gen, mutation_rates))

        num_species = count_number_of_species(new_children)

        #------------#
        # assessment #
        #------------#
        self.update_fitnesses_and_novelty() # run sim

        #-----------#
        # selection #
        #-----------#
        if(c.use_speciation): self.population = new_children # replace parents with new children (mu, lambda)
        else: self.population += new_children # combine parents with new children (the + in mu+lambda)

        self.population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True) # sort the full population by each individual's fitness (from highers to lowest)

        if(not c.use_speciation):
            self.population = tournament_selection(self.population, c, False) # tournament selection (novelty and fitness)
            # self.population = self.population[:num_parents] # truncation

        #----------------#
        # record keeping #
        #----------------#
        # diversity:
        # std_distance, avg_distance, max_diff = calculate_diversity(self.population, self.all_species)
        std_distance, avg_distance, max_diff = calculate_diversity_full(self.population)
        n_nodes = get_avg_number_of_hidden_nodes(self.population)
        n_connections = get_avg_number_of_connections(self.population)
        self.diversity_over_time[self.gen:] = avg_distance
        self.nodes_over_time[self.gen:] = n_nodes
        self.connections_over_time[self.gen:] = n_connections

        # fitness
        if self.population[0].fitness > self.solution_fitness: # if the new parent is the best found so far
            self.solution = self.population[0]                 # update best solution records
            self.solution_fitness = self.population[0].fitness
            self.solution_generation = self.gen
                
        self.fitness_over_time[self.gen:] = self.solution_fitness # record the fitness of the current best over evolutionary time
        self.solutions_over_time.append(copy.deepcopy(self.solution))

        # species
        # adjust the species threshold to get closer to the right number of species
        if(num_species>c.species_target): self.species_threshold+=c.species_threshold_delta
        if(num_species<c.species_target): self.species_threshold-=c.species_threshold_delta
        self.species_threshold = max(0, self.species_threshold)
        self.species_over_time[self.gen:] = num_species
        self.species_threshold_over_time[self.gen:] = self.species_threshold
        champs = get_current_species_champs(self.population, self.all_species)
        self.species_champs_over_time.append(champs) 
        
        self.save_best_network_image()
    
    def get_best(self):
        lowest = max(self.population, key=(lambda k: k.fitness))
        return lowest
    
    def print_best(self):
        best = self.get_best()
        print("Best:", best.id, best.fitness)
        
    def show_best(self):
        print()
        self.print_best()
        self.get_best().start_simulation(False, self.debug_output, True)
        self.save_best_network_image()

    def save_best_network_image(self):
        best = self.get_best()
        visualize_network(self.get_best(), sample=True, save_name=f"best/{time.time()}_{self.gen}_{best.id}.png", extra_text="Generation: " + str(self.gen) + " fit: " + str(best.fitness) + " species: " + str(best.species_id))


def classic_selection_and_reproduction(c, population, all_species, generation_num, mutation_rates):
    new_children = []
    while len(new_children) < c.pop_size:
        # inheritance
        parent1 = np.random.choice(population, size=1)[0] # pick 1 random parent

        #crossover
        if(c.do_crossover):
            parent2 = np.random.choice(get_members_of_species(population, parent1.species_id), size=1)[0] # note, we are allowing parents to crossover with themselves
            child = crossover(parent1, parent2)
        else:
            child = copy.deepcopy(parent1)

        # mutation
        mutate(child, Genome, mutation_rates)

        new_children.extend([child]) # add children to the new_children list

    return new_children


def tournament_selection(population, c, use_adjusted_fitness=False, override_no_elitism=False):
    new_population = []
    if(not override_no_elitism): 
        for i in range(c.population_elitism):
            new_population.append(population[i]) # keep best genomes (elitism)

    # fitness
    while len(new_population) < (1-c.novelty_selection_ratio_within_species)*c.num_parents:
        tournament = np.random.choice(population, size = min(c.tournament_size, len(population)), replace=False)
        if(use_adjusted_fitness):
            tournament = sorted(tournament, key=lambda genome: genome.adjusted_fitness, reverse=True)
        else:
            tournament = sorted(tournament, key=lambda genome: genome.fitness, reverse=True)
        new_population.extend(tournament[:c.tournament_winners])  

    # novelty
    if(c.novelty_selection_ratio_within_species > 0):
        sorted_pop = sorted(population, key=lambda genome: genome.novelty, reverse=True) # sort the full population by each genome's fitness (from highers to lowest)
        while len(new_population) < c.num_parents:
            tournament = np.random.choice(sorted_pop, size = min(c.tournament_size, len(sorted_pop)), replace=False)
            tournament = sorted(tournament, key=lambda genome: genome.novelty, reverse=True)
            new_population.extend(tournament[:c.tournament_winners])  
    
    return new_population  


def calculate_diversity(population, all_species):
    # Compares 1 representative from every species against each other
    reps = []
    for species in all_species:
        members = get_members_of_species(population, species.id)
        if(len(members)<1): continue
        reps.append(np.random.choice(members, 1)[0])
    diffs = []
    for i in reps:
        for j in reps:
            if i== j: continue
            diffs.append(i.genetic_difference(j))

    std_distance = np.std(diffs)
    avg_distance = np.mean(diffs)
    max_diff = np.max(diffs)if(len(diffs)>0) else 0
    return std_distance, avg_distance, max_diff

    
def calculate_diversity_full(population, all_species=None):
    # very slow, compares every genome against every other
    diffs = []
    for i in population:
        for j in population:
            if i== j: continue
            diffs.append(i.genetic_difference(j))

    std_distance = np.std(diffs)
    avg_distance = np.mean(diffs)
    max_diff = np.max(diffs)if(len(diffs)>0) else 0
    return std_distance, avg_distance, max_diff

def update_solution_archive(solution_archive, genome, max_archive_length, novelty_k):
    # genome should already have novelty score
    # update existing novelty scores:
    # for i, archived_solution in enumerate(solution_archive):
    #     solution_archive[i].novelty = get_novelty(solution_archive, genome, novelty_k)
    solution_archive = sorted(solution_archive, reverse=True, key = lambda s: s.novelty)

    if(len(solution_archive) >= max_archive_length):
        if(genome.novelty > solution_archive[-1].novelty):
            # has higher novelty than at least one genome in archive
            solution_archive[-1] = genome # replace least novel genome in archive
    else:
        solution_archive.append(genome)
    return solution_archive

    
def mutate(child, genome, rates):
    prob_mutate_activation, prob_mutate_weight, prob_add_connection, prob_add_node, prob_remove_node, prob_disable_connection, weight_mutation_max, prob_reenable_connection = rates
    
    child.fitness, child.adjusted_fitness = -math.inf, -math.inf # new fitnesses after mutation

    if(np.random.uniform(0,1) < c.prob_random_restart):
        child = genome()
    if(np.random.uniform(0,1) < prob_add_node):
        child.add_node()
    if(np.random.uniform(0,1) < prob_remove_node):
        child.remove_node()
    if(np.random.uniform(0,1) < prob_add_connection):
        child.add_connection()
    if(np.random.uniform(0,1) < prob_disable_connection):
        child.disable_connection()
    # if(np.random.uniform(0,1)< prob_mutate_activation):
    
    child.mutate_activations()
    child.mutate_weights()
