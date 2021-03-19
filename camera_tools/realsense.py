"""
Wrapper to work with Intel RealSense Camera
"""

import numpy as np
import pyrealsense2 as rs

DEPTH_RESOLUTION = (1024, 768)
COLOR_RESOLUTION = (1920, 1080)

class RealSenseCamera():

    def __init__(self):
        self.colorizer = rs.colorizer()
        self.pipeline = rs.pipeline()
        
        config = rs.config()
        config.enable_stream(rs.stream.depth, DEPTH_RESOLUTION[0],
                             DEPTH_RESOLUTION[1], rs.format.z16, 30)
        config.enable_stream(rs.stream.color, COLOR_RESOLUTION[0],
                             COLOR_RESOLUTION[1], rs.format.rgb8, 30)
        self.pipeline.start(config)

    def read_rgb(self):
        while True:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_frame = np.asanyarray(color_frame.get_data())
            return color_frame

    def read_rgbd(self):
        while True:
            frames = self.pipeline.wait_for_frames()

            # Create alignment primitive with color as its target stream:
            align = rs.align(rs.stream.color)
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            color_frame = color_frame.get_data()
            color_frame = np.asanyarray(color_frame)

            #depth_frame = self.colorizer.colorize(depth_frame).get_data()
            depth_frame = np.asanyarray(depth_frame.get_data())

            # Convert images to numpy arrays
            return color_frame, depth_frame
    
    def stop(self):
        self.pipeline.stop()