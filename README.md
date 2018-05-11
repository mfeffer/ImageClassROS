# Image Classification ROS Package

This repo contains the work of the Image Classification team for the 6.834 Grand Challenge. We have specifically created a ROS package that follows a client-server model to classify parts of an image taken by a camera node via a convolutional neural net (CNN).

## Getting Started

The rest of this section will go over parts of the repo and how to build the ROS package.

### Structure

This is a ROS package, so `CMakeLists.txt` and `package.xml` contain information about the package and are used when running `catkin_make` (more on that later). Those should not need to be modified.

The `src` directory is empty, but `scripts` contains our client and server implementations. We use `rospy` and therefore these implementations are in Python. This directory also contains our code and neural net model required for classifying parts of the image that are taken by the camera.

`scripts` additionally contains information about different models our team has explored. Though our ROS pipeline uses a CNN, we also have performed training experiments with a Bag-of-Features (BoF) implementation and included relevant code and documentation in the `bof` subdirectory. We also include similar code and documentation in the `cnn` subdirectory. 

### Prerequisites

Our code assumes usage of ROS kinetic, Python 2, and Keras (with TensorFlow backend). Install these as necessary.

### Installing

First, clone the repo and place the contents in your `catkin` workspace. Then, run `catkin_make` to prepare the package to be used. 

## Usage

###Initialization

After those two setup steps, run each of the following commands in different processes or terminal windows:

Run `roscore` on remote computer (running ROS):

```
roscore
```

Bring robot online to run with camera (assuming turtlebot3):

```
roslaunch turtlebot3_bringup turtlebot3_rpicamera.launch
```

Run server on remote computer (running ROS):

```
rosrun image_class ros_server.py
```

Run client on remote computer (running ROS):

```
rosrun image_class ros_client.py 
```

###Result

The first two steps allow the computer and robot to communicate. The server is a process that runs in an infinite loop waiting for requests containing images from the client (in turn obtained from the robot's camera node) and classifies patches in the image and sends back the results to the client. The client is a transient process that waits for an image from the camera node, sends the image to the server for classification of patches, and upon obtaining the results, it returns and prints them to standard output and exits. (We can change client behavior depending on the needs of other teams such as the Gaussian Processes team; for instance it should not be difficult to publish to a node and/or run continuously.)
