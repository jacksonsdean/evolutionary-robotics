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

    robot_to_show = "best_brain"
    if args.robot is not None:
        robot_to_show = args.robot
    sensors = ["FrontLowerLeg", "BackLowerLeg", "LeftLowerLeg", "RightLowerLeg"]
    data_name = f"sensors_{robot_to_show.lower()}"
    
    footprint = np.zeros((len(data[data_name][sensors[0]]), len(sensors)))
    for robot, robot_data in data.items():
        if not data_name == robot.lower():
            continue
        for sensor, sensor_data in robot_data.items():
            if sensor in sensors:
                y_index = sensors.index(sensor)
                footprint[:, y_index] = sensor_data
           
    steps_to_show = len(footprint)
    if args.steps is not None:
        steps_to_show = int(args.steps)   

    # only the first M values are used
    footprint = footprint[:steps_to_show, :]
    footprint = footprint.T
    
    plt.figure(figsize=(18, 5))
    ax=plt.gca()
    # adjust the y axis scale.
    ax.locator_params('y', nbins=4)
    # Hide major tick labels
    ax.set_yticklabels('')

    # Customize minor tick labels
    ax.set_yticks(np.add(np.arange(len(sensors)) , .5),      minor=True)
    ax.set_yticklabels(["empty"] + sensors, minor=False)

    plt.ylim(ymin=-0.5,ymax=3.5)
    plt.xlabel("Time")
    plt.grid(True, axis='y', which="minor")
    plt.title(f"Touch sensor values for {args.title if args.title else robot_to_show}")
    plt.imshow(footprint, cmap="binary", vmin=-1, vmax=1, aspect="auto", interpolation="none")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run search on the robot controller.')
    parser.add_argument('-f', '--data_file',
                        action='store', help='Data results file.')
    parser.add_argument('-r', '--robot',
                        action='store', help='Robot brain name.')
    parser.add_argument('-s', '--steps',
                        action='store', help='Number of steps to show.')
    parser.add_argument('-t', '--title',
                        action='store', help='Plot title. Defaults to robot name.')
    args = parser.parse_args()
    main(args)
