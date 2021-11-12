#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author     : Hongzhuo Liang
# E-mail     : liang@informatik.uni-hamburg.de
# Description:
# Date       : 05/08/2018 6:04 PM
# File Name  : kinect2grasp.py

import torch
import rospy
from sensor_msgs.msg import PointCloud2
import os
import sys
# import tf
# import moveit_commander
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "../../")
sys.path.append(ROOT_DIR)

from scripts.utils import pointclouds
from scripts.utils import voxelgrid
from scripts.utils.plot import PlotTools

import pcl
from autolab_core import YamlConfig
from dexnet.grasping import RobotGripper
from dexnet.grasping import GpgGraspSamplerPcl

from pyquaternion import Quaternion

from os import path
import time
from scipy.stats import mode
import multiprocessing as mp
from scipy.spatial.transform import Rotation as R

# 自定义消息
from grasp_detect.srv._GraspDetect import GraspDetect, GraspDetectRequest, GraspDetectResponse
from geometry_msgs.msg import PoseStamped

try:
    from mayavi import mlab
except ImportError:
    print("Can not import mayavi")
    mlab = None

sys.path.append(os.path.join(ROOT_DIR, "PointNetGPD"))

# global config:
yaml_config = YamlConfig(os.path.join(ROOT_DIR, "dex-net/test/config.yaml"))
gripper_name = 'robotiq_85'
gripper = RobotGripper.load(gripper_name,
                            os.path.join(ROOT_DIR, "dex-net/data/grippers"))
ags = GpgGraspSamplerPcl(gripper, yaml_config)
value_fc = 0.4  # no use, set a random number
num_grasps = 40
num_workers = 4
max_num_samples = 100
n_voxel = 100

minimal_points_send_to_point_net = 20
marker_life_time = 8

show_bad_grasp = False
save_grasp_related_file = False

show_final_grasp = False
tray_grasp = False
using_mp = False

# number of points put into neural network
model_type = "3class"
if model_type == "100":  # minimal points send for training
    input_points_num = 500
elif model_type == "50":
    input_points_num = 750
elif model_type == "3class":
    input_points_num = 500
else:
    input_points_num = 0


def remove_table_points(points_voxel_, vis=False):
    xy_unique = np.unique(points_voxel_[:, 0:2], axis=0)
    new_points_voxel_ = points_voxel_
    pre_del = np.zeros([1])
    for i in range(len(xy_unique)):
        tmp = []
        for j in range(len(points_voxel_)):
            if np.array_equal(points_voxel_[j, 0:2], xy_unique[i]):
                tmp.append(j)
        print(len(tmp))
        if len(tmp) < 3:
            tmp = np.array(tmp)
            pre_del = np.hstack([pre_del, tmp])
    if len(pre_del) != 1:
        pre_del = pre_del[1:]
        new_points_voxel_ = np.delete(points_voxel_, pre_del, 0)
    print("Success delete [[ {} ]] points from the table!".format(
        len(points_voxel_) - len(new_points_voxel_)))

    if vis:
        p = points_voxel_
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      scale_factor=0.002,
                      color=(1, 0, 0))
        p = new_points_voxel_
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      scale_factor=0.002,
                      color=(0, 0, 1))
        mlab.points3d(0, 0, 0, scale_factor=0.01,
                      color=(0, 1, 0))  # plot 0 point
        mlab.show()
    return new_points_voxel_


def remove_white_pixel(msg, points_, vis=False):
    points_with_c_ = pointclouds.pointcloud2_to_array(msg)
    points_with_c_ = pointclouds.split_rgb_field(points_with_c_)
    r = np.asarray(points_with_c_['r'], dtype=np.uint32)
    g = np.asarray(points_with_c_['g'], dtype=np.uint32)
    b = np.asarray(points_with_c_['b'], dtype=np.uint32)
    rgb_colors = np.vstack([r, g, b]).T
    # rgb = rgb_colors.astype(np.float) / 255
    ind_good_points_ = np.sum(rgb_colors[:] < 210, axis=-1) == 3
    ind_good_points_ = np.where(ind_good_points_ == 1)[0]
    new_points_ = points_[ind_good_points_]
    if vis:
        p = points_
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      scale_factor=0.002,
                      color=(1, 0, 0))
        p = new_points_
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      scale_factor=0.002,
                      color=(0, 0, 1))
        mlab.points3d(0, 0, 0, scale_factor=0.01,
                      color=(0, 1, 0))  # plot 0 point
        mlab.show()
    return new_points_


def get_voxel_fun(points_, n):
    get_voxel = voxelgrid.VoxelGrid(points_, n_x=n, n_y=n, n_z=n)
    get_voxel.compute()
    points_voxel_ = get_voxel.voxel_centers[get_voxel.voxel_n]
    points_voxel_ = np.unique(points_voxel_, axis=0)
    return points_voxel_


def cal_grasp(input_point, cam_pos_):
    points_ = input_point.astype(np.float32)
    remove_white = False
    # if remove_white:
    #     points_ = remove_white_pixel(msg, points_, vis=True)

    ####################### begin voxel points #######################
    n = n_voxel  # parameter related to voxel method
    # gpg improvements, highlights: flexible n parameter for voxelizing.
    points_voxel_ = get_voxel_fun(points_, n)
    if len(points_) < 2000:  # should be a parameter
        while len(points_voxel_) < len(points_) - 15:
            points_voxel_ = get_voxel_fun(points_, n)
            n = n + 100
            rospy.loginfo(
                "the voxel has {} points, we want get {} points".format(
                    len(points_voxel_), len(points_)))

    rospy.loginfo("the voxel has {} points, we want get {} points".format(
        len(points_voxel_), len(points_)))

    points_ = points_voxel_
    remove_points = False
    if remove_points:
        points_ = remove_table_points(points_, vis=True)
    point_cloud = pcl.PointCloud(points_)
    norm = point_cloud.make_NormalEstimation()
    norm.set_KSearch(30)  # critical parameter when calculating the norms
    normals = norm.compute()
    surface_normal = normals.to_array()
    surface_normal = surface_normal[:, 0:3]
    vector_p2cam = cam_pos_ - points_
    vector_p2cam = vector_p2cam / np.linalg.norm(vector_p2cam, axis=1).reshape(
        -1, 1)
    tmp = np.dot(vector_p2cam, surface_normal.T).diagonal()
    angel = np.arccos(np.clip(tmp, -1.0, 1.0))
    wrong_dir_norm = np.where(angel > np.pi * 0.5)[0]
    tmp = np.ones([len(angel), 3])
    tmp[wrong_dir_norm, :] = -1
    surface_normal = surface_normal * tmp
    select_point_above_table = 0.010
    #  modify of gpg: make it as a parameter. avoid select points near the table.
    points_for_sample = points_[np.where(
        points_[:, 2] > select_point_above_table)[0]]
    if len(points_for_sample) == 0:
        rospy.loginfo(
            "Can not seltect point, maybe the point cloud is too low?")
        return [], points_, surface_normal
    yaml_config['metrics']['robust_ferrari_canny']['friction_coef'] = value_fc
    if not using_mp:
        rospy.loginfo("Begin cal grasps using single thread, slow!")
        grasps_together_ = ags.sample_grasps(point_cloud,
                                             points_for_sample,
                                             surface_normal,
                                             num_grasps,
                                             max_num_samples=max_num_samples,
                                             show_final_grasp=show_final_grasp)
    else:
        # begin parallel grasp:
        rospy.loginfo("Begin cal grasps using parallel!")

        def grasp_task(num_grasps_, ags_, queue_):
            ret = ags_.sample_grasps(point_cloud,
                                     points_for_sample,
                                     surface_normal,
                                     num_grasps_,
                                     max_num_samples=max_num_samples,
                                     show_final_grasp=show_final_grasp)
            queue_.put(ret)

        queue = mp.Queue()
        num_grasps_p_worker = int(num_grasps / num_workers)
        workers = [
            mp.Process(target=grasp_task,
                       args=(num_grasps_p_worker, ags, queue))
            for _ in range(num_workers)
        ]
        [i.start() for i in workers]

        grasps_together_ = []
        for i in range(num_workers):
            grasps_together_ = grasps_together_ + queue.get()
        rospy.loginfo("Finish mp processing!")
    rospy.loginfo("Grasp sampler finish, generated {} grasps.".format(
        len(grasps_together_)))
    return grasps_together_, points_, surface_normal


def check_collision_square(grasp_bottom_center,
                           approach_normal,
                           binormal,
                           minor_pc,
                           points_,
                           p,
                           way="p_open"):
    approach_normal = approach_normal.reshape(1, 3)
    approach_normal = approach_normal / np.linalg.norm(approach_normal)
    binormal = binormal.reshape(1, 3)
    binormal = binormal / np.linalg.norm(binormal)
    minor_pc = minor_pc.reshape(1, 3)
    minor_pc = minor_pc / np.linalg.norm(minor_pc)
    matrix_ = np.hstack([approach_normal.T, binormal.T, minor_pc.T])
    grasp_matrix = matrix_.T
    points_ = points_ - grasp_bottom_center.reshape(1, 3)
    tmp = np.dot(grasp_matrix, points_.T)
    points_g = tmp.T
    use_dataset_py = True
    if not use_dataset_py:
        if way == "p_open":
            s1, s2, s4, s8 = p[1], p[2], p[4], p[8]
        elif way == "p_left":
            s1, s2, s4, s8 = p[9], p[1], p[10], p[12]
        elif way == "p_right":
            s1, s2, s4, s8 = p[2], p[13], p[3], p[7]
        elif way == "p_bottom":
            s1, s2, s4, s8 = p[11], p[15], p[12], p[20]
        else:
            raise ValueError('No way!')
        a1 = s1[1] < points_g[:, 1]
        a2 = s2[1] > points_g[:, 1]
        a3 = s1[2] > points_g[:, 2]
        a4 = s4[2] < points_g[:, 2]
        a5 = s4[0] > points_g[:, 0]
        a6 = s8[0] < points_g[:, 0]

        a = np.vstack([a1, a2, a3, a4, a5, a6])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True
    # for the way of pointGPD/dataset.py:
    else:
        width = ags.gripper.hand_outer_diameter - 2 * ags.gripper.finger_width
        x_limit = ags.gripper.hand_depth
        z_limit = width / 4
        y_limit = width / 2
        x1 = points_g[:, 0] > 0
        x2 = points_g[:, 0] < x_limit
        y1 = points_g[:, 1] > -y_limit
        y2 = points_g[:, 1] < y_limit
        z1 = points_g[:, 2] > -z_limit
        z2 = points_g[:, 2] < z_limit
        a = np.vstack([x1, x2, y1, y2, z1, z2])
        points_in_area = np.where(np.sum(a, axis=0) == len(a))[0]
        if len(points_in_area) == 0:
            has_p = False
        else:
            has_p = True

    vis = False
    if vis:
        p = points_g
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      scale_factor=0.002,
                      color=(0, 0, 1))
        p = points_g[points_in_area]
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      scale_factor=0.002,
                      color=(1, 0, 0))
        p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]),
                                np.array([0, 1, 0]))
        mlab.points3d(p[:, 0],
                      p[:, 1],
                      p[:, 2],
                      scale_factor=0.005,
                      color=(0, 1, 0))
        mlab.show()

    return has_p, points_in_area, points_g


def collect_pc(grasp_, pc):
    """
    grasp_bottom_center, normal, major_pc, minor_pc
    """
    grasp_num = len(grasp_)
    grasp_ = np.array(grasp_)
    grasp_ = grasp_.reshape(-1, 5,
                            3)  # prevent to have grasp that only have number 1
    grasp_bottom_center = grasp_[:, 0]
    approach_normal = grasp_[:, 1]
    binormal = grasp_[:, 2]
    minor_pc = grasp_[:, 3]

    in_ind_ = []
    in_ind_points_ = []
    p = ags.get_hand_points(np.array([0, 0, 0]), np.array([1, 0, 0]),
                            np.array([0, 1, 0]))
    for i_ in range(grasp_num):
        has_p, in_ind_tmp, points_g = check_collision_square(
            grasp_bottom_center[i_], approach_normal[i_], binormal[i_],
            minor_pc[i_], pc, p)
        in_ind_.append(in_ind_tmp)
        in_ind_points_.append(points_g[in_ind_[i_]])
    return in_ind_, in_ind_points_


def check_hand_points_fun(real_grasp_, points):
    ind_points_num = []
    for i in range(len(real_grasp_)):
        grasp_bottom_center = real_grasp_[i][4]
        approach_normal = real_grasp_[i][1]
        binormal = real_grasp_[i][2]
        minor_pc = real_grasp_[i][3]
        local_hand_points = ags.get_hand_points(np.array([0, 0, 0]),
                                                np.array([1, 0, 0]),
                                                np.array([0, 1, 0]))
        has_points_tmp, ind_points_tmp = ags.check_collision_square(
            grasp_bottom_center, approach_normal, binormal, minor_pc, points,
            local_hand_points, "p_open")
        ind_points_num.append(len(ind_points_tmp))
    print(ind_points_num)
    if (len(ind_points_num) == 0):
        print("len(ind_points_num) == 0")
        return
    # file_name = "./generated_grasps/real_points/" + str(
    #     np.random.randint(300)) + str(len(real_grasp_)) + ".npy"
    # np.save(file_name, np.array(ind_points_num))


def remove_grasp_outside_tray(grasps_, points_):
    x_min = points_[:, 0].min()
    x_max = points_[:, 0].max()
    y_min = points_[:, 1].min()
    y_max = points_[:, 1].max()
    valid_grasp_ind_ = []
    for i in range(len(grasps_)):
        grasp_bottom_center = grasps_[i][4]
        approach_normal = grasps_[i][1]
        major_pc = grasps_[i][2]
        hand_points_ = ags.get_hand_points(grasp_bottom_center,
                                           approach_normal, major_pc)
        finger_points_ = hand_points_[[1, 2, 3, 4, 9, 10, 13, 14], :]
        # aa = points_[:, :2] - finger_points_[0][:2]  # todo： work of remove outside grasp not finished.

        # from IPython import embed;embed()
        a = finger_points_[:, 0] < x_min
        b = finger_points_[:, 0] > x_max
        c = finger_points_[:, 1] < y_min
        d = finger_points_[:, 1] > y_max
        if np.sum(a) + np.sum(b) + np.sum(c) + np.sum(d) == 0:
            valid_grasp_ind_.append(i)
    grasps_inside_ = [grasps_[i] for i in valid_grasp_ind_]
    rospy.loginfo(
        "gpg got {} grasps, after remove grasp outside tray, {} grasps left".
        format(len(grasps_), len(grasps_inside_)))
    return grasps_inside_


class GraspDetectorServer(object):
    def __init__(self, service_name) -> None:
        super().__init__()
        '''
        Service(name, service_class, handler, buff_size=DEFAULT_BUFF_SIZE, error_handler=None)
        name 话题命名
        service_class　服务端接收的数据类型
        handler　服务端调用的callback函数
        '''
        self.server_ = rospy.Service(
            service_name,
            GraspDetect,
            self.serverCallBack,
        )

        model_path = "/home/waha/catkin_research_ws/src/pointnet-gpd-ros/data/pointnetgpd_3class.model"

        use_gpu = True

        # self.model = torch.load(model_path, map_location="cpu")
        # self.model.device_ids = [0]
        # print("load model {}".format(model_path))

        if use_gpu:
            self.model = torch.load(model_path,
                                    map_location="cuda:{}".format(0))
            torch.cuda.set_device(0)
            self.model = self.model.cuda()

        rospy.loginfo('Service startup completed')

    def serverCallBack(self, request: GraspDetectRequest):
        cam_pos = [0, 0, 1]
        input_points = None
        if (request.pcd_path is not ""):
            pcl_cloud = pcl.load(request.pcd_path)
            input_points = pointclouds.pcl_cloud_to_xyz_array(pcl_cloud)
        else:
            input_points = pointclouds.pointcloud2_to_xyz_array(
                request.point_cloud)

        plot = PlotTools()
        plot.addPointcloud(input_points)
        real_good_grasp = []
        real_bad_grasp = []
        real_score_value = []

        repeat = 1  # speed up this try 10 time is too time consuming

        real_grasp, points, normals_cal = cal_grasp(input_points, cam_pos)
        if tray_grasp:
            real_grasp = remove_grasp_outside_tray(real_grasp, points)

        check_grasp_points_num = True  # evaluate the number of points in a grasp
        check_hand_points_fun(real_grasp,
                              points) if check_grasp_points_num else 0

        in_ind, in_ind_points = collect_pc(real_grasp, points)
        if save_grasp_related_file:
            np.save("./generated_grasps/points.npy", points)
            np.save("./generated_grasps/in_ind.npy", in_ind)
            np.save("./generated_grasps/real_grasp.npy", real_grasp)
            np.save("./generated_grasps/cal_norm.npy", normals_cal)
        score = []  # should be 0 or 1
        score_value = []  # should be float [0, 1]
        ind_good_grasp = []
        ind_bad_grasp = []
        rospy.loginfo("Begin send grasp into pointnet, cal grasp score")
        for ii in range(len(in_ind_points)):
            if in_ind_points[ii].shape[0] < minimal_points_send_to_point_net:
                rospy.loginfo(
                    "Mark as bad grasp! Only {} points, should be at least {} points."
                    .format(in_ind_points[ii].shape[0],
                            minimal_points_send_to_point_net))
                score.append(0)
                score_value.append(0.0)
                if show_bad_grasp:
                    ind_bad_grasp.append(ii)
            else:
                predict = []
                grasp_score = []
                for _ in range(repeat):
                    # 采样点，大于设定数目，不重复采样，否则允许重复采样
                    if len(in_ind_points[ii]) >= input_points_num:
                        points_modify = in_ind_points[ii][np.random.choice(
                            len(in_ind_points[ii]),
                            input_points_num,
                            replace=False)]
                    else:
                        points_modify = in_ind_points[ii][np.random.choice(
                            len(in_ind_points[ii]),
                            input_points_num,
                            replace=True)]
                    if_good_grasp, grasp_score_tmp = self.test_network(
                        self.model.eval(), points_modify)
                    predict.append(if_good_grasp.item())
                    grasp_score.append(grasp_score_tmp)

                predict_vote = mode(predict)[0][
                    0]  # vote from all the "repeat" results.
                grasp_score = np.array(grasp_score)
                if model_type == "3class":  # the best in 3 class classification is the last column, third column
                    which_one_is_best = 2  # should set as 2
                else:  # for two class classification best is the second column (also the last column)
                    which_one_is_best = 1  # should set as 1
                score_vote = np.mean(grasp_score[np.where(
                    predict == predict_vote)][:, 0, which_one_is_best])
                score.append(predict_vote)
                score_value.append(score_vote)

                if score[ii] == which_one_is_best:
                    ind_good_grasp.append(ii)
                else:
                    if show_bad_grasp:
                        ind_bad_grasp.append(ii)
        print("Got {} good grasps, and {} bad grasps".format(
            len(ind_good_grasp),
            len(in_ind_points) - len(ind_good_grasp)))

        if len(ind_good_grasp) != 0:
            real_good_grasp = [real_grasp[i] for i in ind_good_grasp]
            real_score_value = [score_value[i] for i in ind_good_grasp]
            if show_bad_grasp:
                real_bad_grasp = [real_grasp[i] for i in ind_bad_grasp]
        # end of grasp detection
        # get sorted ind by the score values
        sorted_value_ind = list(
            index for index, item in sorted(enumerate(real_score_value),
                                            key=lambda item: item[1],
                                            reverse=True))
        # sort grasps using the ind
        sorted_real_good_grasp = [real_good_grasp[i] for i in sorted_value_ind]
        real_good_grasp = sorted_real_good_grasp
        # get the sorted score value, from high to low
        real_score_value = sorted(real_score_value, reverse=True)

        # print(real_good_grasp, real_score_value)
        print(ind_bad_grasp)
        response = GraspDetectResponse()

        for i in range(len(real_grasp)):
            print(i)
            pose = PoseStamped()
            print(real_grasp[i], score_value[i])
            center = real_grasp[i][4]
            pose.pose.position.x = center[0]
            pose.pose.position.y = center[1]
            pose.pose.position.z = center[2]

            approach = real_grasp[i][1]
            close = real_grasp[i][2]
            vertical = real_grasp[i][3]
            r = R.from_matrix([[approach[0], close[0], vertical[0]],
                               [approach[1], close[1], vertical[1]],
                               [approach[2], close[2], vertical[2]]])
            quat = r.as_quat()

            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            response.grasp_pose.append(pose)

            plot.addGrasp(center, approach, close)
        plot.show()
        rospy.loginfo(
            " Publishing grasp pose to rviz using marker array and good grasp pose"
        )
        # rospy.sleep(10)
        return response

    def test_network(self, model_, local_pc):
        local_pc = local_pc.T
        local_pc = local_pc[np.newaxis, ...]
        local_pc = torch.FloatTensor(local_pc)
        if True:
            local_pc = local_pc.cuda()
        output, _ = model_(local_pc)  # N*C
        output = output.softmax(1)
        pred = output.data.max(1, keepdim=True)[1]
        output = output.cpu()
        return pred[0], output.data.numpy()


if __name__ == '__main__':
    """
    definition of gotten grasps:

    grasp_bottom_center = grasp_[0]
    approach_normal = grasp_[1]
    binormal = grasp_[2]
    """

    rospy.init_node('pointnet_gpd_ros_server', anonymous=True)

    server = GraspDetectorServer("/grasp_detecter/get_pose")

    rospy.spin()
