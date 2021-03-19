#! /usr/bin/env python3
"""
Capture focal plane depth map
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'camera_tools/'))

import cv2
import numpy as np
import matplotlib.pyplot as plt

import camera_tools.calibrate as calibrate
import camera_tools.perspective as perspective
import camera_tools.realsense as rs

FOCAL_DEPTH_FILE = 'etc/focal_depth.npy'

PROP_FILE = 'camera_tools/etc/camera_config.json'
PERSPECTIVE_FILE = 'camera_tools/etc/perspective.json'

# Load Camera Calibration and Perspective Calibration Files
camera_matrix, dist_coeffs = calibrate.load_camera_props(PROP_FILE)
mapx, mapy = calibrate.get_undistort_maps(camera_matrix, dist_coeffs)
m, max_width, max_height = perspective.load_perspective(PERSPECTIVE_FILE) 

# Task 1: Capture Registered RGBD Data
stream = rs.RealSenseCamera()

color, depth = stream.read_rgbd()
stream.stop()

color = calibrate.undistort_image(color, mapx, mapy)
depth = calibrate.undistort_image(depth, mapx, mapy)

color = cv2.warpPerspective(color, m, (max_width, max_height))
depth = cv2.warpPerspective(depth, m, (max_width, max_height))

color = cv2.resize(color, rs.COLOR_RESOLUTION)
depth = cv2.resize(depth, rs.COLOR_RESOLUTION)

plt.figure()
plt.imshow(color)
plt.title("Focal Plane - RGB")
plt.axis('off')
plt.savefig('fig/focal_rgb.png', dpi=100, bbox_inches='tight')


plt.figure()
plt.imshow(depth)
plt.title("Focal Plane - Depth")
plt.axis('off')
plt.savefig('fig/focal_depth.png', dpi=100, bbox_inches='tight')



np.save(FOCAL_DEPTH_FILE, depth)