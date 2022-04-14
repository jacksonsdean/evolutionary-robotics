import math
import random
import numpy as np

class Species:
    def __init__(self, _id) -> None:
        self.id = _id
        self.avg_adjusted_fitness = -math.inf
        self.avg_fitness = -math.inf
        self.allowed_offspring = 0
        self.population_count = 0
        self.last_fitness= self.avg_adjusted_fitness
        self.last_improvement = 0
        self.current_champ = None
    
    def update(self, global_adjusted_fitness, members, gen, stagnation_threshold, total_pop):
        self.avg_adjusted_fitness = np.mean([i.adjusted_fitness for i in members])
        self.avg_fitness =  np.mean([i.fitness for i in members])
        if(self.avg_fitness > self.last_fitness):
            self.last_improvement = gen
        self.last_fitness = self.avg_fitness

        # Every species is assigned a potentially different number of offspring in proportion to the sum of ad-
        # justed fitnesses of its member organisms. Species then reproduce by first eliminating
        # the lowest performing members from the population. The entire population is then
        # replaced by the offspring of the remaining organisms in each species.

        if(gen- self.last_improvement>= stagnation_threshold):
            self.allowed_offspring = 0
        else:
            try:
                # nk = (Fk/Ftot)*P 
                self.allowed_offspring = int(round(self.population_count * (self.avg_adjusted_fitness / global_adjusted_fitness)))
                if self.allowed_offspring < 0: self.allowed_offspring = 0
            except ArithmeticError:
                print(f"error while calc allowed_offspring: pop:{self.population_count} fit:{self.avg_adjusted_fitness} glob: {global_adjusted_fitness}")
            except ValueError:
                print(f"error while calc allowed_offspring: pop:{self.population_count} fit:{self.avg_adjusted_fitness} glob: {global_adjusted_fitness}")



def get_adjusted_fitness_of_species(population, species):
    return np.mean([i.adjusted_fitness for i in population if i.species_id == species])
def get_fitness_of_species(population, species):
    return np.mean([i.fitness for i in population if i.species_id == species])
def count_members_of_species(population, species):
    return len(get_members_of_species(population, species))
def get_members_of_species(population, species):
    return [ind for ind in population if ind.species_id == species]
def count_number_of_species(population):
    species = []
    for ind in population:
        if(ind.species_id not in species):
            species.append(ind.species_id)
    return len(species)

def get_current_species_champs(population, all_species):
    for sp in all_species:
        members = get_members_of_species(population, sp.id)
        sp.population_count = len(members)
        if(sp.population_count==0): continue
        members= sorted(members, key=lambda x: x.fitness, reverse=True)
        sp.current_champ = members[0]
    return [sp.current_champ for sp in all_species if (sp.population_count>0 and sp.current_champ is not None)]


def assign_species(all_species, population, threshold, SpeciesClass):
    reps = {}
    for s in all_species:
        species_pop = get_members_of_species(population, s.id)
        s.population_count = len(species_pop)
        if(s.population_count<1): continue
        reps[s.id] = np.random.choice(species_pop, 1)[0]
        
    
    # The Genome Loop:
    for g in population:
        # – Take next genome g from P
        placed = False
        species = list(range(len(all_species)))
        random.shuffle(species)
        
        # – The Species Loop:
        for s_index in species:
            s = all_species[s_index]
            # ·get next species s from S
            species_pop = get_members_of_species(population, s.id)
            s.population_count = len(species_pop)
            if(s.population_count<1): continue
            if(g.species_comparision(reps[s.id], threshold)):
            # if(g.species_comparision(species_pop[0], threshold)):
                # ·If g is compatible with s, add g to s
                g.species_id = s.id
                placed = True
                break
        if(not placed):
            # ∗If all species in S have been checked, create new species and place g in it
            new_id = len(all_species)
            all_species.append(SpeciesClass(new_id))
            reps[new_id] = g
            g.species_id = new_id

def normalize_species_offspring(all_species, c):
    # Normalize the number of allowed offspring per species so that the total is close to pop_size
    total_offspring = np.sum([s.allowed_offspring for s in all_species])
    target_children = c.pop_size
    target_children -= c.population_elitism # first children will be most fit from last gen 
    if(total_offspring == 0): total_offspring = 1 # TODO FIXME (total extinction)

    norm = c.pop_size / total_offspring
    
    for sp in all_species:
        try:
            sp.allowed_offspring = int(round(sp.allowed_offspring * norm))
        except ValueError as e:
            print(f"unexpected value during species offspring normalization, ignoring: {e} offspring: {sp.allowed_offspring} norm:{norm}")
            continue

    return norm

def normalize_species_offspring_exact(all_species, pop_size):
    # Jackson's method (always exact pop_size)
    # if there are not enough offspring, assigns extras to top (multiple) species,
    # if there are too many, takes away from worst (multiple) species
    total_offspring = np.sum([s.allowed_offspring for s in all_species])
    adj = 1 if total_offspring<pop_size else -1
    sorted_species = sorted(all_species, key=lambda x: x.avg_adjusted_fitness, reverse=(total_offspring<pop_size))
    while(total_offspring!=pop_size):
        for s in sorted_species:
            if(s.population_count == 0 or s.allowed_offspring == 0): continue
            s.allowed_offspring+=adj
            total_offspring+=adj
            print("adj=", adj)
            if(total_offspring==pop_size): break
