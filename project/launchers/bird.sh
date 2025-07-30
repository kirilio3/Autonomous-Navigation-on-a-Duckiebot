#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch publisher
rosrun my_package homography_bird.py

# wait for app to end
dt-launchfile-join