import json

import numpy as np
    
from util import *
from neat_genome import *

class Experiment:
    def __init__(self, config, name, num_runs=1) -> None:
        self.config = config
        self.name = name
        self.results = []
        self.num_runs = num_runs
        self.experiment_results =np.zeros((num_runs, config.total_generations))
        self.diversity_results = np.zeros((num_runs, config.total_generations))
        self.species_results =   np.zeros((num_runs, config.total_generations))
        self.threshold_results = np.zeros((num_runs, config.total_generations))
        self.nodes_results = np.zeros((num_runs, config.total_generations))
        self.connections_results = np.zeros((num_runs, config.total_generations))
        self.gens_to_find_solution = [math.inf] * num_runs
        self.solutions_results = []
        self.gens_to_converge = []
        self.species_champs_results = []
        self.me_maps = []
        self.current_run = 0
            
    def found_solution(self, generation):
        self.gens_to_find_solution[self.current_run] = generation
    def record_results(self, fitness_over_time, diversity_over_time, solutions_over_time, species_over_time, species_threshold_over_time, nodes_over_time, connections_over_time, gens_to_converge, species_champs_over_time, me_map):
        self.experiment_results[self.current_run] = fitness_over_time
        self.diversity_results[self.current_run] = diversity_over_time
        self.species_results[self.current_run] = species_over_time
        self.threshold_results[self.current_run] = species_threshold_over_time
        self.nodes_results[self.current_run] = nodes_over_time
        self.connections_results[self.current_run] = connections_over_time
        self.solutions_results.append(solutions_over_time)
        self.gens_to_converge.append(gens_to_converge)
        self.species_champs_results.append(species_champs_over_time)
        self.current_run+=1
        self.me_maps.append(me_map)

    def show_results(self, visualize_disabled_cxs=False):
        self.show_target_and_trained_images()
        self.show_training_images()
        self.show_best_network(visualize_disabled_cxs)

    def show_best_network(self, visualize_disabled_cxs=False):
        visualize_network(get_best_solution_from_all_runs(self.solutions_results)[0], color_mode=self.config.color_mode,
            visualize_disabled=visualize_disabled_cxs,
            sample=True, sample_point=[.5,.5])


    def show_species_champs(self, run=-1, gen=-1,):
        print(f"Run {run}, Generation {gen}: {len(self.species_champs_results[run][gen])} species")

    def plot_results(self, do_bootstrap):
        plot_mean_and_bootstrapped_ci_over_time(self.experiment_results, [self.experiment_results], self.name, "Generation", "Fitness", plot_bootstrap=do_bootstrap)
        # plot diversity 
        plot_mean_and_bootstrapped_ci_over_time(self.diversity_results, [self.diversity_results], self.name, "Generation", "Diversity", plot_bootstrap=do_bootstrap)
        # plot species
        plot_mean_and_bootstrapped_ci_over_time(self.species_results, [self.species_results], self.name, "Generation", "N Species", plot_bootstrap=do_bootstrap)
        plot_mean_and_bootstrapped_ci_over_time(self.threshold_results, [self.threshold_results], self.name, "Generation", "Species Threshold", plot_bootstrap=do_bootstrap)
        
        # Topology
        plot_mean_and_bootstrapped_ci_over_time([self.nodes_results, self.connections_results], [self.nodes_results, self.connections_results], ["Nodes", "Connections"], "Generation", "Topological Units", plot_bootstrap=do_bootstrap)


    def generate_gif_best(self, res = [-1,-1]):
        best, run_index = get_best_solution_from_all_runs(self.solutions_results)
        if(-1 in res): 
            if(isinstance(self.config.train_image, str)):
                res = self.config.classification_image_size
            else:
                res = self.config.train_image.shape
        imgs = []
        for solution in self.solutions_results[run_index]:
            img = solution.get_image(res[0], res[1], self.config.color_mode)
            if not any(np.allclose(img, s) for s in imgs):
                imgs.append(img)

        for i in range(3): imgs.append(imgs[-1]) # extend end frame
        # images_to_gif(imgs, f"saved_experiment_{time.time()}.gif", fps=2)

    def generate_gif_family_tree(self, res = [-1,-1]):
        imgs = []
        if(-1 in res): 
            if(isinstance(self.config.train_image, str)):
                res = self.config.classification_image_size
            else:
                res = self.config.train_image.shape
        for solution in self.get_best_family_tree():
            img = solution.get_image(res[0], res[1], self.config.color_mode)
            if not any(np.allclose(img, s) for s in imgs):
                imgs.append(img)

        for i in range(3): imgs.append(imgs[-1]) # extend end frame
        # images_to_gif(imgs, f"saved_experiment_{time.time()}.gif", fps=2)

    def get_best_family_tree(self):
        parent, index = get_best_solution_from_all_runs(self.solutions_results)
        family_tree = []
        while parent!=None:
            family_tree.append(parent)
            parent = parent.more_fit_parent
        return family_tree
    
    def save_json(self, filename):
        with open(filename, 'w+') as outfile:
            strng = self.to_json()
            outfile.write(strng)
            outfile.close()
    def load_saved(self, filename):
        with open(filename, 'r') as infile:
            data = json.load(infile)
            self.from_json(data)
            infile.close()
    def from_json(self, json_dict):
        self.config = self.config.to_json()
        self.__dict__ = json_dict
        self.experiment_results  = np.array(self.experiment_results) 
        self.diversity_results   = np.array(self.diversity_results)
        self.species_results     = np.array(self.species_results)
        self.threshold_results   = np.array(self.threshold_results)
        self.nodes_results       = np.array(self.nodes_results)
        self.connections_results = np.array(self.connections_results)
        self.config = Config.CreateFromJson(self.config)
        self.recover_solutions()

    def to_json(self):
        self.experiment_results  = self.experiment_results.tolist()
        self.diversity_results   = self.diversity_results.tolist()
        self.species_results     = self.species_results.tolist()
        self.threshold_results   = self.threshold_results.tolist()
        self.nodes_results       = self.nodes_results.tolist()
        self.connections_results = self.connections_results.tolist()
        self.config = self.config.to_json()
        self.make_solutions_serializable()

        json_string = json.dumps(self.__dict__, sort_keys=True, indent=4)

        self.experiment_results  = np.array(self.experiment_results) 
        self.diversity_results   = np.array(self.diversity_results)
        self.species_results     = np.array(self.species_results)
        self.threshold_results   = np.array(self.threshold_results)
        self.nodes_results       = np.array(self.nodes_results)
        self.connections_results = np.array(self.connections_results)
        self.config = Config.CreateFromJson(self.config)
        self.recover_solutions()
        
        
        return json_string

        
    def LoadFromSave(filename):
        config = Config()
        experiment = Experiment(config, "loading")
        experiment.load_saved(filename)
        return experiment



def plot_generations_to_converge(experiments):
    X = [e.name for e in experiments]
    Y = [np.mean(e.gens_to_converge) for e in experiments]
    plt.bar(X, Y)
    plt.title("Generations to converge")
    plt.xlabel("Experiment")
    plt.ylabel("Generations")
    plt.show()
    