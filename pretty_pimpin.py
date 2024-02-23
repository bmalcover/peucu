import bagpy
from bagpy import bagreader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from sensor_msgs.msg import Image

import pyrealsense2 as rs
import cv2
import matplotlib.pyplot as plt

src = "data/p_0001/2022-05-19/bag/p_0001_19052022_00.bag"

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, src, repeat_playback=False)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    # Start streaming from file
    profile = pipeline.start(config)
    
    # Create colorizer object
    colorizer = rs.colorizer()

    # Streaming loop
    number = 0
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        
        if not frames:
            break

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
   
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_min = 0.11 #meter
        depth_max = 1.0 #meter

        depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        print("depth_intrin, ", type(depth_intrin))
        print(depth_intrin)
        color_intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        print("color_intrin")
        print(color_intrin)
        depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
        color_to_depth_extrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))

        print("depth_to_color_extrin")
        print(depth_to_color_extrin)

        print("color_to_depth_extrin")
        print(color_to_depth_extrin)

        color_points = [
            [400, 150],
            [560, 150],
            [560, 260],
            [400, 260]
        ]

        color_resized = cv2.resize(color_image, (depth_color_image.shape[1], depth_color_image.shape[0]))

        for color_point in color_points:

            depth_point_ = rs.rs2_project_color_pixel_to_depth_pixel(
                        depth_frame.get_data(), depth_scale,
                        depth_min, depth_max,
                        depth_intrin, color_intrin,
                        depth_to_color_extrin,
                        color_to_depth_extrin,
                        color_point)

            color_image = cv2.circle(color_image, (int(color_point[1]), int(color_point[0])), radius=4, color=(0, 255, 255), thickness=-1)
            depth_color_image = cv2.circle(depth_color_image, (int(depth_point_[1]), int(depth_point_[0])), radius=4, color=(255, 0, 0), thickness=-1)



        cv2.imwrite("data2/{:03d}".format(number) + ".png", color_image)
        cv2.imwrite("data2/{:03d}".format(number) +"_d.png", depth_color_image)
        number += 1

        break

finally:
    pass