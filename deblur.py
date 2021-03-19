import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'camera_tools/'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

from pypher.pypher import psf2otf
from fspecial import fspecial_gaussian_2d

import camera_tools.calibrate as calibrate
import camera_tools.perspective as perspective
import camera_tools.realsense as rs
import camera_tools.fullscreen.fullscreen as fs
from simulate_blur import filter_rgb

TILE_SCALE = 120
E_PSF_DIMENSIONS = (8, 8)
BLUR_DEPTH_COEFFICIENT = 1 / 120

FOCAL_DEPTH_FILE = 'etc/focal_depth.npy'
PROP_FILE = 'camera_tools/etc/camera_config.json'
PERSPECTIVE_FILE = 'camera_tools/etc/perspective.json'

# Load Camera Calibration and Perspective Calibration Files
camera_matrix, dist_coeffs = calibrate.load_camera_props(PROP_FILE)
mapx, mapy = calibrate.get_undistort_maps(camera_matrix, dist_coeffs)
m, max_width, max_height = perspective.load_perspective(PERSPECTIVE_FILE)

# Load Focal Plane Depth
focal_plane = np.load(FOCAL_DEPTH_FILE)

stream = rs.RealSenseCamera()
screen = fs.FullScreen(0)

# Load original image
original = io.imread('img/test.JPEG') / 255
original = cv2.resize(original, (1920, 1080))

SNR = np.mean(original) / 0.01
TILE_HEIGHT = original.shape[0] // TILE_SCALE
TILE_WIDTH = original.shape[1] // TILE_SCALE
while True:
    # Task 1: Capture Registered and Rectified RGBD Data
    color, depth = stream.read_rgbd()

    color = calibrate.undistort_image(color, mapx, mapy)
    depth = calibrate.undistort_image(depth, mapx, mapy)

    color = cv2.warpPerspective(color, m, (max_width, max_height))
    depth = cv2.warpPerspective(depth, m, (max_width, max_height))

    color = cv2.resize(color, rs.COLOR_RESOLUTION)
    depth = cv2.resize(depth, rs.COLOR_RESOLUTION)

    focal_offsets = np.abs(depth.astype(np.int16) - focal_plane.astype(np.int16))
    focal_offsets = focal_offsets.astype(np.uint8)
    #focal_offsets = cv2.bilateralFilter(focal_offsets, 100, 100, 100)


    # Task 2: Estimate Spatially Varying PSFs with tiles
    tile_sigmas = cv2.resize(focal_offsets,
                             (focal_offsets.shape[1] // TILE_SCALE,
                              focal_offsets.shape[0] // TILE_SCALE))
    tile_sigmas = tile_sigmas * BLUR_DEPTH_COEFFICIENT


    # Task 3: Apply spatially-varying convolutions to tiles
    precond = np.zeros(original.shape)
    for row in range(0, tile_sigmas.shape[0]):
        for col in range(0, tile_sigmas.shape[1]):
            top = row * TILE_SCALE
            bot = top + TILE_SCALE
            lef = col * TILE_SCALE
            rig = lef + TILE_SCALE

            tile_sigma = tile_sigmas[row, col]

            e_psf = fspecial_gaussian_2d(E_PSF_DIMENSIONS, tile_sigma)
            e_otf = psf2otf(e_psf, (TILE_SCALE, TILE_SCALE))

            # Wiener Pre-Filtering
            wiener_filt = e_otf.conj() / (np.abs(e_otf ** 2) + 1 / SNR)
            precond[top:bot, lef:rig] = filter_rgb(original[top:bot, lef:rig],
                                                   wiener_filt)
    precond = np.clip(precond, 0, 1)
    precond = (precond * 255).astype(np.uint8)
    screen.imshow(cv2.cvtColor(precond, cv2.COLOR_BGR2RGB))
    
    if cv2.waitKey(1) & 255 == ord('q'):
        break

    # Uncomment below to save a single shot image
    #  plt.figure(figsize=(30, 30))
    #  plt.imshow(color)
    #  plt.title('RGB Image of Scene')
    #  plt.axis('off')
    #  plt.savefig('fig/rgb_image.png', dpi=100, bbox_inches='tight')
    #
    #  plt.figure(figsize=(30, 30))
    #  plt.imshow(depth)
    #  plt.title('Depth Image of Scene')
    #  plt.axis('off')
    #  plt.savefig('fig/depth_image.png', dpi=100, bbox_inches='tight')
    #
    #  plt.figure(figsize=(30, 30))
    #  plt.imshow(focal_offsets)
    #  plt.title('Offsets from Focal Plane')
    #  plt.savefig('fig/offsets.png', dpi=100, bbox_inches='tight')
    #
    #  plt.figure(figsize=(30, 30))
    #  plt.imshow(tile_sigmas)
    #  plt.title('Per-Tile Sigmas')
    #  plt.savefig('fig/sigmas.png')
    #
    #  plt.figure(figsize=(30, 30))
    #  plt.imshow(precond)
    #  plt.title('Preconditioned Image')
    #  plt.savefig('fig/precond.png', dpi=100, bbox_inches='tight')
    #
    #  color, depth = stream.read_rgbd()
    #  color = calibrate.undistort_image(color, mapx, mapy)
    #  color = cv2.warpPerspective(color, m, (max_width, max_height))
    #  color = cv2.resize(color, rs.COLOR_RESOLUTION)
    #  plt.figure(figsize=(30, 30))
    #  plt.imshow(color)
    #  plt.title('Projected Pre-Conditioned Image')
    #  plt.axis('off')
    #  plt.savefig('fig/projected.png', dpi=100, bbox_inches='tight')
    #  break
    

cv2.destroyAllWindows()
stream.stop()
