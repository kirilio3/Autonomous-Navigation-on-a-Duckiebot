#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch publisher
rosrun my_package lane_follow_3d.py

# wait for app to end
dt-launchfile-join