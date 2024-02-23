import numpy as np
import pyrealsense2 as rs


def toMetric3d(p, depth, i_depth):

    x_d, y_d = p

    p3d = np.zeros(3)  # x,y,z

    p3d[0] = (x_d - i_depth.ppx) * depth.get_distance(x_d, y_d) / i_depth.fx
    p3d[1] = (y_d - i_depth.ppy) * depth.get_distance(x_d, y_d) / i_depth.fy
    p3d[2] = depth.get_distance(x_d, y_d)

    return p3d


def reprojection(p3d, extrinsics, i_color):

    """
    P3D' = R.P3D + T
    P2D_rgb.x = (P3D'.x * fx_rgb / P3D'.z) + cx_rgb
    P2D_rgb.y = (P3D'.y * fy_rgb / P3D'.z) + cy_rgb
    """

    R = np.asarray(extrinsics.rotation).reshape((3, 3))
    T = np.asarray(extrinsics.translation)

    p3d1 = np.dot(R, p3d) + T
    p2d_rgb = np.zeros(2)

    p2d_rgb[0] = ((p3d1[0] * i_color.fx) / p3d1[2]) + i_color.ppx
    p2d_rgb[1] = ((p3d1[1] * i_color.fy) / p3d1[2]) + i_color.ppy

    return p2d_rgb