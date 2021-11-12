from operator import truediv
import os, sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mayavi.mlab as mlab
from numpy.__config__ import show


class PlotTools(object):
    def __init__(self) -> None:
        super().__init__()
        self.pointcloud_vec = []
        self.grasp_vec = []

    def clear(self):
        self.pointcloud_vec.clear()
        self.grasp_vec.clear()

    def addPointcloud(self, points):
        self.pointcloud_vec.append(np.array(points))

    def addGrasp(self, position, approach, closure):
        self.grasp_vec.append(
            [np.array(position),
             np.array(approach),
             np.array(closure)])

    def show(self):
        for points in self.pointcloud_vec:
            PlotTools.plotPointCloud(points, show=False)

        for grasp in self.grasp_vec:
            PlotTools.plotGrasp(np.array(grasp[0]),
                                np.array(grasp[1]),
                                np.array(grasp[2]),
                                show=False)

        mlab.show()

    @staticmethod
    def plotPointCloud(points, show=True):

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        mlab.points3d(x, y, z, scale_factor=0.005)
        if show:
            mlab.show()

    @staticmethod
    def plotGrasp(position, approach, closure, points=None, show=True):
        if points is not None:
            PlotTools.plotPointCloud(points, show=False)

        grasp_color = (0, 1, 0)
        bottom_position = position - 0.07 * approach
        line = np.vstack((bottom_position, position))

        mlab.plot3d(line[:, 0],
                    line[:, 1],
                    line[:, 2],
                    color=grasp_color,
                    representation='wireframe',
                    line_width=5,
                    tube_radius=0.002,
                    tube_sides=6)
        # ax.plot(line[:, 0], line[:, 1], line[:, 2], c='r', color=grasp_color)

        left_position = position - 0.045 * closure
        right_position = position + 0.045 * closure
        line = np.vstack((left_position, right_position))
        mlab.plot3d(line[:, 0],
                    line[:, 1],
                    line[:, 2],
                    color=grasp_color,
                    representation='wireframe',
                    line_width=5,
                    tube_radius=0.002,
                    tube_sides=6)
        # ax.plot(line[:, 0], line[:, 1], line[:, 2], c='r', color=grasp_color)

        left_top_position = left_position + 0.06 * approach
        line = np.vstack((left_position, left_top_position))
        mlab.plot3d(line[:, 0],
                    line[:, 1],
                    line[:, 2],
                    color=grasp_color,
                    representation='wireframe',
                    line_width=5,
                    tube_radius=0.002,
                    tube_sides=6)
        # ax.plot(line[:, 0], line[:, 1], line[:, 2], c='r', color=grasp_color)

        right_top_position = right_position + 0.06 * approach
        line = np.vstack((right_position, right_top_position))
        mlab.plot3d(line[:, 0],
                    line[:, 1],
                    line[:, 2],
                    color=grasp_color,
                    representation='wireframe',
                    line_width=5,
                    tube_radius=0.002,
                    tube_sides=6)
        # ax.plot(line[:, 0], line[:, 1], line[:, 2], c='r', color=grasp_color)

        if show:
            mlab.show()
        # plt.show()
