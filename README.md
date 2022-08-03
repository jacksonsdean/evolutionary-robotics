# UVM CS206 - Evolutionary Robotics
## Jackson Dean
Class assignments and final project


# Instructions
Install the requirements using `pip install -r requirements.txt`

## Single run
A single run of NEAT for a quadruped controller can be done with:
`python search.py`

Change the algorithm to HyperNEAT with `python search.py --alg=hyperneat`

Other command line options for number of generations, runs, population size, etc. can be seen with `python search.py --help`

## Experimentation
To run an experiment comparing different algorithms or hyperparameters, create or modify one of the JSON files in the `experiments` directory. Most fields in the `Constants` class (`constants.py`) can be changed. Run the experiment with `python search.py --f=<path_to_json>`. 

For example, to run an experiment comparing NEAT to HyperNEAT with a population size of 20, for 100 generations, and 3 experimental trials, use:
`python search.py -f "experiments/neat_vs_hyperneat.json" -r 3`
Results will be saved to `experiments/neat_vs_hyperneat_results.json` and shown after the run.

## Visualization
To visualize the results of an experiment, run `python analyze_experiment.py -f <path_to_results.json>`.

Optional bootstrapped confidence intervals can be shown on line plots with `-b`.

To show a simulation of the best brain from a condition, run `python analyze_experiment.py -f <path_to_results.json> -s <condition_name>`.

For example, to show the final results of this project and visualize the NEAT controller, run:
`python analyze_experiment.py -f experiments/final_results.json -s neat`



### Disclaimer
This project is a work in progress. The code is not production ready, and may contain bugs.

Most development/testing has been on Windows so not sure if it works on Mac or Linux.

## Credits
This project uses a custom version of [Pyrosim](https://github.com/jbongard/pyrosim) and was developed for Dr. Joshua Bongard's Evolutionary Robotics course at the University of Vermont. Check out the course's open [subreddit](https://www.reddit.com/r/ludobots/) for more information. Video examples of this project are available on [YouTube](https://www.youtube.com/channel/UCc9_Sf3XFUdJOO1ZvAcdZsg).