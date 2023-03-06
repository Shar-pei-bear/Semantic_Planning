# Semantic_Planning
Bridging planning and semantic SLAM in an virtual environment

## Simulation Environment Setup
Install the 0.2.1 version of [Habitat-sim](https://github.com/facebookresearch/habitat-sim/tree/v0.2.1) and [Habitat-lab](https://github.com/facebookresearch/habitat-lab/tree/v0.2.1).

## Build instruction
Navigate to the project root directory, run:

`catkin init`

`catkin build -j4`

## Running instruction
### start the simulation

In the root directory, run:

`source devel/setup.bash`

`python3 HabitatSimulation.py`

### start the planner
In the root directory, run:

`source devel/setup.bash`

`roslaunch habitatlab_rtabmap.launch`
