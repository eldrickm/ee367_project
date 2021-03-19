# EE 367 Final Project - Dynamic Defocus Deblurring with Depth Data

## Author

[eldrickm](https://github.com/eldrickm)


## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [References](#references)


## Overview

Files:

- `fspecial.py`
  Library to generate Gaussian kernels. Given as starter code
  during the course

- `capture_focal_plane.py`
  Basic helper script to capture and save a depth image from the RealSense

- `simulate_blur.py`
  Simulates defocus blur and deblurring on test images.

- `deblur.py`
  Runs dynamic deblurring on the projector-camera system

Directories:

- `img/`
  Contains input test images

- `fig/`
  Contains output images as a result of the scripts

- `etc/`
  Contains saved data such as focal plane depth

- `camera_tools`
  Contains a library to interface with the RealSense, as well as
  scripts to calibrate cameras and calculate perspective homography for
  projector-camera systems.


## Setup

  This codebase is written in Python 3.x.
  You will need to install the relevant dependencies in the scripts.
  Most should be fulfilled by a standard Anaconda install.

  You may need to `pip install pypher` for the `psf2otf` function.
  You may need to `pip install opencv-contrib-python` for the
  tools used in `camera_tools`.

  For physical setup, you will need a projector and an Intel RealSense L515.
  `fullscreen` should automatically identify your projector screen
  as the secondary screen for most laptop setups.

  The projector's projectable area should be entirely within view of the
  L515's field of view.


## Usage

### Simulated Blurring

Follow the steps below to simulate blurring and pre-conditioning techniques.

### Step 1: Simulate Blurring

  In the top level folder, run `./simulate_blur.py`.
  You can modify the argument to main to change the input image.
  I suggest opening this file in an IDE such as Spyder to
  inspect intermediate values and facilitate easy data exploration.

  You can uncomment the lower blocks to project some test patterns.


### Dynamic Deblurring

Follow the steps below to use the full dynamic deblurring pipeline
for a physical projector-camera system setup

### Step 1: Camera Calibration

  In `camera_tools` run `./calibrate.py`.
  A ChArUco board will be projected out by the projector.
  The RGB camera will capture 50 frames of this board and will
  return the camera intrinsics and distortion coefficients in
  `camera_tools/etc` as `camera_config.json`

### Step 2: Projector-Camera Perspective Homography

  In `camera_tools` run `./perspective.py`.
  A white rectangle will be projected out by the projector.
  The RGB camera will capture an image of this frame and a
  homography will be saved in `camera_tools/etc` as `perspective.json`

### Step 3: Capture Focal Plane

  In the top level folder, run `./capture_focal_plane.py`.
  The RealSense camera will capture a depth image and it will be saved in
  `etc/` as `focal_depth.npy`

### Step 4: Dynamic Deblurring

  In the top level folder, run `./deblur.py`
  This will begin the dynamic deblurring process.
  You can comment in the lower block to save a single frame for debugging.
  You can press `q` to quit.



# References

Full academic references can be found in the report.
Included here are codebases that were referenced or used in this
implementation.

[IntelRealSense/librealsense](https://github.com/IntelRealSense/librealsense)
- Reference examples to work with the L515 RealSense camera

[elerac/fullscreen](https://github.com/elerac/fullscreen)
- Displaying images in fullscreen on projector
- Source contained in `fullscreen/`

[dclemmon/projection_mapping](https://github.com/dclemmon/projection_mapping)
- Initial implementation of camera calibration and perspective transformation
- Modified source files in `camera_tools/`
