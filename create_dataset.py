import numpy as np
import pyrealsense2 as rs
import cv2
import os
import utils


def imagesfromfile(src):

    user = src.split("/")[1]
    directory_path = "data2/" + user

    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print("The directory exists.")
    else:
        os.makedirs(directory_path)

    try:
        # Create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, src, repeat_playback=False)

        # Configure the pipeline to stream the depth stream
        # Change these parameters according to the recorded bag file resolution
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

        # Start streaming from file
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)

        align_to = rs.stream.color
        align = rs.align(align_to)

        # Streaming loop
        number = 0
        while True:

            try:
                frames = pipeline.wait_for_frames(1000)
            except RuntimeError:
                print("END!")
                break
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            cv2.imwrite("data2/" + user + "/color/{:03d}".format(number) + ".png", color_image)
            cv2.imwrite("data2/" + user + "/depth/{:03d}".format(number) + ".png", depth_image)
            number += 1



            print("Creating image number {:03d}".format(number))

    finally:
        print("Work done!")


if __name__ == "__main__":
    src = "data/p_0001/2022-05-19/bag/p_0001_19052022_00.bag"

    imagesfromfile(src)
