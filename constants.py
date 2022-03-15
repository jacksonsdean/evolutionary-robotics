import numpy as np

gravity = 0, 0, -9.8

simulation_fps = 240
simulation_length = 300

frontMotorAmplitude = np.pi/4.
frontMotorFreq = 10.
frontMotorPhaseOffset = np.pi*80.0/180.0

backMotorAmplitude = np.pi/4.
backMotorFreq = 10.0
backMotorPhaseOffset = 0.0

front_max_force = 100.
back_max_force = 100.

num_gens = 1
pop_size = 1

num_motor_neurons = 4
num_sensor_neurons = 5
