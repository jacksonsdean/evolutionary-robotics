import argparse
import numpy as np
import matplotlib.pyplot as plt
import json

# %%
def main(args):
    filename = "data/sensor_values.json"
    if args.data_file:
        filename = args.data_file
    with open(filename) as f:
        data = json.load(f)

    if args.sensor is None:
        for robot, robot_data in data.items():
            for sensor, sensor_data in robot_data.items():
                plt.plot(sensor_data, label=f"{robot}_{sensor}")
    else:
        for robot, robot_data in data.items():
            for sensor, sensor_data in robot_data.items():
                if sensor.lower() == args.sensor.lower():
                    plt.plot(sensor_data, label=f"{robot}_{sensor}")
        
        
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run search on the robot controller.')
    parser.add_argument('-f', '--data_file',
                        action='store', help='Data results file.')
    parser.add_argument('-s', '--sensor',
                        action='store', help='Sensor to plot, all if omitted.')
    args = parser.parse_args()
    main(args)
