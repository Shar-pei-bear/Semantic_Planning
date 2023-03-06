import cv2
import habitat_sim
import numpy as np
from habitat_sim.utils import viz_utils as vut
from cv_bridge import CvBridge
import random
import os
from sample_objects import sample_object_state
from habitat_config import *
from gripper import *
from map import TopDownMap
from tracking import *
from ex import HabitatObstacleCSpace, CSpaceObstacleProgram
from visualize import *
import argparse
import rospy
from sensor_msgs.msg import Image, CameraInfo
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes, \
    RangeBearing, RangeBearings, RobotCommand
from numpy.linalg import inv, norm
import quaternion as qt
import copy
import pickle
from scipy.stats import dirichlet
from scipy.io import savemat
from habitat.utils.visualizations import maps


def match_color(object_mask, target_color, tolerance=3):
    min_val = target_color - tolerance
    max_val = target_color + tolerance
    match_region = (object_mask >= min_val) & (object_mask <= max_val)
    if match_region.sum() != 0:
        return match_region
    else:
        return None


class Environment:
    def __init__(self, args):
        self.sim = None
        self.obj_attr_mgr = None
        self.rigid_obj_mgr = None
        self.prim_attr_mgr = None
        self.stage_attr_mgr = None

        self.sel_file_obj_handle = ""
        self.sel_prim_obj_handle = ""
        self.sel_asset_handle = ""

        self.show_video = args.display
        self.display = args.display
        self.make_video = args.make_video
        self.make_video = True

        dir_path = './'
        self.data_path = os.path.join(dir_path, "data")
        output_directory = "examples/tutorials/interactivity_output/"
        self.output_path = os.path.join(dir_path, output_directory)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        self.id2category = {}
        self.ids = set()
        self.categories = set()

        self.sim_settings = make_default_settings()
        self.sim_settings[
            "scene"] = "./data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
        self.sim_settings["sensor_pitch"] = 0
        self.sim_settings["sensor_height"] = 0.6
        self.sim_settings["color_sensor_3rd_person"] = True
        self.sim_settings["depth_sensor_1st_person"] = True
        self.sim_settings["semantic_sensor_1st_person"] = True

        self.make_simulator_from_settings()
        self.build_widget_ui()

        default_nav_mesh_settings = habitat_sim.NavMeshSettings()
        default_nav_mesh_settings.set_defaults()

        inflated_nav_mesh_settings = habitat_sim.NavMeshSettings()
        inflated_nav_mesh_settings.set_defaults()
        inflated_nav_mesh_settings.agent_radius = 0.2
        inflated_nav_mesh_settings.agent_height = 1.5

        recompute_successful = self.sim.recompute_navmesh(self.sim.pathfinder,
                                                          inflated_nav_mesh_settings)
        if not recompute_successful:
            print("Failed to recompute navmesh!")

        seed = 24
        random.seed(seed)
        self.sim.seed(seed)
        np.random.seed(seed)

        cube_handle = self.obj_attr_mgr.get_template_handles("cube")[0]
        sphere_handle = \
            self.obj_attr_mgr.get_template_handles("uvSphereSolid")[0]
        cheezit_handle = self.obj_attr_mgr.get_template_handles("cheezit")[0]
        # load the locobot_merged asset
        locobot_template_handle = \
            self.obj_attr_mgr.get_file_template_handles("locobot")[0]

        # load a selected target object and place it on the NavMesh
        self.objs = []
        self.target_objs = []
        self.target_class = "towel"
        self.objs.append(self.rigid_obj_mgr.add_object_by_template_handle(
            self.sel_file_obj_handle))
        self.objs.append(self.rigid_obj_mgr.add_object_by_template_handle(
            cheezit_handle))

        # add robot object to the scene with the agent/camera SceneNode attached
        self.locobot_obj = self.rigid_obj_mgr.add_object_by_template_handle(
            locobot_template_handle, self.sim.agents[0].scene_node)
        # set the agent's body to kinematic since we will be updating position manually
        self.locobot_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        # create and configure a new VelocityControl structure
        # Note: this is NOT the object's VelocityControl, so it will not be consumed automatically in sim.step_physics
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True

        # reset observations and robot state
        self.locobot_obj.translation = self.sim.pathfinder.get_random_navigable_point()

        # for i in range(10):
        #     print(self.sim.pathfinder.get_random_navigable_point())

        # kitchen towel position
        # 3.830780e+00 1.035790e+00 - 2.834797e-04

        # [-8.5, 0.07244661, -0.9758241]
        # [-7.256092,    0.07244661,  1.9062117]
        # [-10.206394,     0.07244661, - 1.7745175]
        # [-5.1125464,   0.07244661,  0.20282874]
        # [-3.1971934,   0.07244661, - 1.2343142]
        #[-0.07918948,  0.07244661, - 0.6778215]



        start_positions = ([-8.5, 0.07244661, -0.9758241],
                           [-7.256092,    0.07244661,  1.9062117],
                           [-10.206394,     0.07244661, - 1.7745175],
                           [-5.1125464,   0.07244661,  0.20282874],
                           [-3.1971934,   0.07244661, - 1.2343142],
                           [-0.07918948,  0.07244661, - 0.6778215],
                           [3.0265996,   0.07244661, - 2.6372855],
                           [-0.02636409,  0.07244661, - 0.21025816],
                           [2.2829885,   0.07244661, - 1.4528316],
                           [1.2926915,  0.07244661, 0.70288],
                           [3.5013633,  0.07244661, 0.71318156],
                           [1.8259766,   0.07244661, - 3.8485773])


        self.locobot_obj.translation = np.asarray(start_positions[2]) # 8

        print('the initial position of the robot is',
              self.locobot_obj.translation)
        y_rotation = mn.Quaternion.rotation(mn.Rad(0.5 * math.pi),
                                            mn.Vector3(0, 1.0, 0))

        # y_rotation = mn.Quaternion.rotation(mn.Rad(0.4 * 2 * math.pi),
        #                                     mn.Vector3(0, 1.0, 0))

        self.locobot_obj.rotation = y_rotation * self.locobot_obj.rotation
        print(self.locobot_obj.rotation)
        self.observations = []
        self.log_info = []

        self.path1 = habitat_sim.ShortestPath()
        self.path2 = habitat_sim.ShortestPath()
        self.path3 = None
        self.path4 = None
        self.path5 = habitat_sim.MultiGoalShortestPath()

        self.find_path()

        self.print_scene_recur()

        recompute_successful = self.sim.recompute_navmesh(self.sim.pathfinder,
                                                          default_nav_mesh_settings)
        if not recompute_successful:
            print("Failed to recompute navmesh 2!")

        self.vis_objs = []
        self.gripper = ObjectGripper(self.sim,
                                     self.locobot_obj.root_scene_node,
                                     np.array([0.0, 0.6, 0.0]))
        self.continuous_path_follower = ContinuousPathFollower(
            self.sim, self.path3, self.locobot_obj.root_scene_node,
            waypoint_threshold=0.4
        )

        self.show_waypoint_indicators = False
        self.step_freq = 30
        self.time_step = 1.0 / self.step_freq

        self.set_up_visualization()

        map_resolution = 544
        max_episode_steps = 5000
        fov = 90
        visibility_dist = 3
        self.semantic_map = TopDownMap(self.sim, map_resolution,
                                       max_episode_steps, fov,
                                       visibility_dist, self.sim.agents[0])
        self.semantic_map.reset_metric()
        self.metric_map = self.semantic_map.return_original_map()
        self.metric_map[self.metric_map == 2] = 0
        self.metric_map = self.metric_map * 100

        with open('metric.npy', 'wb') as f:
            np.save(f, self.metric_map)

        self.path5.requested_start = self.locobot_obj.translation
        short_length = np.infty
        requested_ends = []
        for obj_id, obj in enumerate(self.target_objs):
            contours = find_sightings(self.metric_map, obj, self.sim, obj_id)
            for contour in contours:
                contour = np.squeeze(contour)
                for pt in contour:
                    realworld_x, realworld_y = maps.from_grid(
                        pt[0],
                        pt[1],
                        self.metric_map.shape[0:2],
                        sim=self.sim,
                    )
                    map_pose = self.sim.pathfinder.snap_point([realworld_y,  0.07244661, realworld_x])
                    if map_pose[0] != map_pose[0]:
                        continue
                    requested_ends.append(map_pose)


            self.path5.requested_ends = requested_ends
            found_path = self.sim.pathfinder.find_path(self.path5)
            if found_path and self.path5.geodesic_distance < short_length:
                short_length = self.path5.geodesic_distance

        # last_key_point = self.path5.points[-2]
        # last_mile = np.linalg.norm(last_key_point - self.path5.requested_end)
        print('shortest path is', short_length)
        # if last_mile < 3:
        #     print('shortest path is', short_length - last_mile)
        # else:
        #     print('shortest path is', short_length - 3)
        print('bound is', self.sim.pathfinder.get_bounds())
        self.metric_map = cv2.cvtColor(self.metric_map, cv2.COLOR_GRAY2BGR)
        map_with_target_objects = draw_target_objects(self.metric_map, self.target_objs, self.sim)
        map_with_target_objects = draw_start_positions(map_with_target_objects, start_positions, self.sim)
        self.way_points = []


        cv2.imwrite('debug.png', map_with_target_objects)

        self.nClass = len(self.categories) + len(self.objs)
        #self.nClass = len(self.categories)
        self.max_time = 30

        self.bridge = CvBridge()

        self.rate = rospy.Rate(self.step_freq)  # hz
        self.seq = 1

        self.camera_info_message = CameraInfo()

        self.camera_info_message.header.frame_id = "camera_rgb_frame"

        self.fx = self.sim_settings["width"] / 2
        self.fy = self.sim_settings["height"] / 2
        self.cx = self.sim_settings["width"] / 2
        self.cy = self.sim_settings["height"] / 2

        self.K = np.asarray([[self.fx, 0, self.cx], [0, self.fx, self.cy], [0, 0, 1]])

        self.camera_info_message.height = self.sim_settings["height"]
        self.camera_info_message.width = self.sim_settings["width"]
        self.camera_info_message.distortion_model = "plumb_bob"
        self.camera_info_message.K = [self.fx, 0, self.cx, 0, self.fx, self.cy, 0, 0, 1]
        self.camera_info_message.D = [0, 0, 0, 0, 0]
        self.camera_info_message.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
        self.camera_info_message.P = [self.fx, 0, self.cx, 0, 0, self.fx, self.cy, 0, 0, 0, 1.0, 0]

        self.alpha_constant = 1.1 #1.06
        self.total_path_length = 0

        self.image_pub = rospy.Publisher("/rgb/image", Image, queue_size=10)
        self.depth_pub = rospy.Publisher("/depth/image", Image, queue_size=10)
        self.semantic_pub = rospy.Publisher("/rgb/bounding_boxes",
                                            BoundingBoxes, queue_size=10)
        self.camera_info_pub = rospy.Publisher("/rgb/camera_info", CameraInfo,
                                               queue_size=10)
        self.range_pub = rospy.Publisher("/range_bearing", RangeBearings, queue_size=10)

        self.command_sub = rospy.Subscriber("/robot_command", RobotCommand,
                                           self.cmd_callback)
        self.mutex_lock_on = False
        self.time_index = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('/home/bear/Github/habitat-lab/fog_2.mp4', fourcc, 30, (1074, 544))

        rospy.on_shutdown(self.save_data)

    def cmd_callback(self, msg):
        self.vel_control.linear_velocity = mn.Vector3(0, 0, msg.linear_vel)
        self.vel_control.angular_velocity = mn.Vector3(0, msg.angular_vel, 0)

    def find_path(self):
        # get the shortest path to the object from the agent position
        found_path = False

        space = HabitatObstacleCSpace(self.sim,
                                      self.locobot_obj.translation[1])

        while not found_path:
            if not sample_object_state(
                self.sim, self.objs[0], from_navmesh=True,
                maintain_object_up=True,
                max_tries=1000, stage_attr_mgr=self.stage_attr_mgr
            ):
                print("Couldn't find an initial object placement. Aborting.")
                break

            if not sample_object_state(
                self.sim, self.objs[1], from_navmesh=True,
                maintain_object_up=True,
                max_tries=1000, stage_attr_mgr=self.stage_attr_mgr
            ):
                print("Couldn't find an initial object placement. Aborting.")
                break

            self.objs[0].translation = np.asarray(
                [-9.015947, 0.07244661, -0.9758241])

            self.path1.requested_start = self.locobot_obj.translation
            self.path1.requested_end = self.objs[0].translation
            self.path2.requested_start = self.path1.requested_end
            # path2.requested_end = sim.pathfinder.get_random_navigable_point()
            self.path2.requested_end = self.objs[1].translation
            found_path = self.sim.pathfinder.find_path(
                self.path1) and self.sim.pathfinder.find_path(self.path2)

        program = CSpaceObstacleProgram(space, self.locobot_obj.translation,
                                        self.objs[0].translation)
        found_path3 = program.run()
        self.path3 = program.produce_path()

        program = CSpaceObstacleProgram(space, self.objs[0].translation,
                                        self.objs[1].translation)
        found_path4 = program.run()
        self.path4 = program.produce_path()

        found_path = found_path3 and found_path4
        print('object 1 translation is', self.objs[1].translation)
        print('path 3 is', self.path3.points)
        print('path 4 is', self.path4.points)
        if not found_path:
            print("Could not find path to object, aborting!")

    def make_simulator_from_settings(self):
        cfg = make_cfg(self.sim_settings)
        # clean-up the current simulator instance if it exists
        if self.sim is not None:
            self.sim.close()
        # initialize the simulator
        self.sim = habitat_sim.Simulator(cfg)
        self.sim.config.sim_cfg.allow_sliding = True
        # Managers of various Attributes templates
        self.obj_attr_mgr = self.sim.get_object_template_manager()
        self.obj_attr_mgr.load_configs(
            str(os.path.join(self.data_path, "objects/example_objects")))
        self.obj_attr_mgr.load_configs(
            str(os.path.join(self.data_path, "objects/locobot_merged")))
        self.prim_attr_mgr = self.sim.get_asset_template_manager()
        self.stage_attr_mgr = self.sim.get_stage_template_manager()
        # Manager providing access to rigid objects
        self.rigid_obj_mgr = self.sim.get_rigid_object_manager()

    # Builds widget-based UI components
    def build_widget_ui(self):
        # Construct DDLs and assign event handlers
        # All file-based object template handles
        file_obj_handles = self.obj_attr_mgr.get_file_template_handles()
        prim_obj_handles = self.obj_attr_mgr.get_synth_template_handles()
        prim_asset_handles = self.prim_attr_mgr.get_template_handles()

        self.sel_file_obj_handle = file_obj_handles[0]
        self.sel_prim_obj_handle = prim_obj_handles[0]
        self.sel_asset_handle = prim_asset_handles[0]

        return

    def set_up_visualization(self):
        if self.show_waypoint_indicators:
            for vis_obj in self.vis_objs:
                self.rigid_obj_mgr.remove_object_by_id(vis_obj.object_id)
            self.vis_objs = setup_path_visualization(
                self.continuous_path_follower,
                self.obj_attr_mgr,
                self.rigid_obj_mgr)

    def print_scene_recur(self, limit_output=10):
        scene = self.sim.semantic_scene
        print(
            f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
        print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

        # 'floor' 'void' '' 'wall' 'ceiling'
        forbiden_classes = ['floor', 'void', '', 'wall', 'ceiling', 'misc', 'picture', 'curtain']

        file = open("/home/bear/Github/habitat-lab/ground_truth.txt", "w")
        for obj in scene.objects:
            category = obj.category.name()
            id = int(obj.id.split("_")[-1])
            if id == 0:
                print('the category of 1 id is: ', category)
            if category not in forbiden_classes:
                id = int(obj.id.split("_")[-1])
                self.id2category[id] = category
                self.ids.add(id)
                self.categories.add(category)

                if category == self.target_class:
                    self.target_objs.append(obj)
                file.write(str(id) +  ' ' + str(obj.aabb.center[2]) + ' ' + str(obj.aabb.center[0]) + ' ' + category + '\n')
        file.close()

        self.ids = list(self.ids)
        self.categories = list(self.categories)
        print('category is ', self.categories)
        self.categories = ['towel', 'objects', 'lighting', 'stool', 'counter',
                           'door', 'clothes', 'appliances', 'furniture',
                           'shelving', 'bed', 'blinds', 'table', 'cabinet',
                           'shower', 'chair', 'chest_of_drawers', 'tv_monitor',
                           'toilet', 'mirror', 'sofa', 'cushion', 'sink',
                           ]

    def run(self):
        while not rospy.is_shutdown():
            previous_rigid_state = self.locobot_obj.rigid_state
            target_rigid_state = self.vel_control.integrate_transform(
                self.time_step, previous_rigid_state
            )

            # snap rigid state to navmesh and set state to object/agent
            end_pos = self.sim.step_filter(
                previous_rigid_state.translation,
                target_rigid_state.translation
            )

            self.locobot_obj.translation = end_pos
            self.locobot_obj.rotation = target_rigid_state.rotation

            dist_moved_before_filter = (
                target_rigid_state.translation - previous_rigid_state.translation
            ).dot()
            dist_moved_after_filter = (
                end_pos - previous_rigid_state.translation).dot()

            # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
            # collision _didn't_ happen. One such case is going up stairs.  Instead,
            # we check to see if the the amount moved after the application of the filter
            # is _less_ than the amount moved before the application of the filter
            EPS = 1e-5
            collided = (
                           dist_moved_after_filter + EPS) < dist_moved_before_filter
            metric = self.semantic_map.update_metric()

            if self.time_index % 2 == 1:
                self.out.write(draw_target_objects(maps.colorize_topdown_map(metric["map"], metric["fog_of_war_mask"]), self.target_objs, self.sim))

            self.total_path_length += np.sqrt(dist_moved_after_filter)

            # run any dynamics simulation
            self.sim.step_physics(self.time_step)
            self.pub_messages()
            # self.observations.append(observation)
            self.time_index += 1
            self.rate.sleep()

    def run_rotate(self):
        y_rotation = mn.Quaternion.rotation(mn.Rad(2 * math.pi / 900),
                                            mn.Vector3(0, 1.0, 0))
        for i in range(900):
            self.locobot_obj.rotation = y_rotation * self.locobot_obj.rotation
            # run any dynamics simulation
            self.sim.step_physics(self.time_step)
            self.pub_messages()
            # self.observations.append(observation)
            self.rate.sleep()

    def run_goal(self):
        start_time = self.sim.get_world_time()
        while (self.continuous_path_follower.progress < 1 and
               self.sim.get_world_time() - start_time < self.max_time
               and not rospy.is_shutdown()):

            self.continuous_path_follower.update_waypoint()
            if self.show_waypoint_indicators:
                self.vis_objs[
                    0].translation = self.continuous_path_follower.waypoint

            if self.locobot_obj.object_id < 0:
                print("locobot_id " + str(self.locobot_obj.object_id))
                break

            previous_rigid_state = self.locobot_obj.rigid_state

            # set velocities based on relative waypoint position/direction
            track_waypoint(
                self.continuous_path_follower.waypoint,
                previous_rigid_state,
                self.vel_control,
                dt=self.time_step,
            )

            # manually integrate the rigid state
            target_rigid_state = self.vel_control.integrate_transform(
                self.time_step, previous_rigid_state
            )

            # snap rigid state to navmesh and set state to object/agent
            end_pos = self.sim.step_filter(
                previous_rigid_state.translation,
                target_rigid_state.translation
            )

            self.locobot_obj.translation = end_pos
            self.locobot_obj.rotation = target_rigid_state.rotation

            self.way_points.append([])

            # Check if a collision occured
            dist_moved_before_filter = (
                target_rigid_state.translation - previous_rigid_state.translation
            ).dot()
            dist_moved_after_filter = (
                end_pos - previous_rigid_state.translation).dot()

            # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
            # collision _didn't_ happen. One such case is going up stairs.  Instead,
            # we check to see if the the amount moved after the application of the filter
            # is _less_ than the amount moved before the application of the filter
            EPS = 1e-5
            collided = (
                           dist_moved_after_filter + EPS) < dist_moved_before_filter

            self.gripper.sync_states()
            # run any dynamics simulation
            self.sim.step_physics(self.time_step)

            self.pub_messages()
            # self.observations.append(observation)
            self.rate.sleep()

    def pub_messages(self):
        # start_time = time()
        # metric = self.semantic_map.update_metric()
        # visible_metric_map = self.metric_map * metric["fog_of_war_mask"]
        # visible_metric_map += (1 - metric["fog_of_war_mask"]) * 50
        #
        # top_down_map = maps.colorize_draw_agent_and_fit_to_height(
        #     metric, 544)
        # fog_of_war_mask = np.expand_dims(metric["fog_of_war_mask"],
        #                                  axis=2)
        # top_down_map = top_down_map * fog_of_war_mask + (
        #     1 - fog_of_war_mask) * 255
        # # fog_of_war_mask = np.expand_dims(metric["fog_of_war_mask"], axis=2)
        # top_down_map = draw_objects(top_down_map, [self.objs[0], self.objs[1]],
        #                             metric["fog_of_war_mask"], self.sim)

        observation = self.sim.get_sensor_observations()

        if self.time_index % 5 == 1:
            video_frame = {"color_sensor_3rd_person": np.concatenate(
                (observation["color_sensor_3rd_person"],
                 observation["color_sensor_1st_person"]), axis=1)}
            self.observations.append(video_frame)

        # rgb_img = Img.fromarray(top_down_map, mode="RGB")
        # rgba_img = rgb_img.convert('RGBA')
        # output_im = np.concatenate((observation[
        #                                 "color_sensor_3rd_person"],
        #                             np.array(rgba_img)), axis=1)
        # observation["color_sensor_3rd_person"] = output_im

        range_sigma = np.sqrt(8e-04)
        bearing_sigma = np.sqrt(0.01)

        range_msg = RangeBearings()
        range_bearings = []

        # rgb_img = observation["color_sensor_1st_person"]

        Twr = np.identity(4)
        Twr[0:3, 0:3] = np.asarray(self.locobot_obj.rotation.to_matrix())
        Twr[0:3, 3] = self.locobot_obj.translation

        Trw = inv(Twr)
        Rrw = Trw[0:3, 0:3]
        trw = Trw[0:3, 3]

        Twc = np.identity(4)
        qua = self.sim.agents[0].get_state().sensor_states[
            'color_sensor_1st_person'].rotation
        Twc[0:3, 0:3] = np.matmul(qt.as_rotation_matrix(qua),
                                  np.asarray(
                                      [[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
        Twc[0:3, 3] = self.sim.agents[0].get_state().sensor_states[
            'color_sensor_1st_person'].position

        Tcw = inv(Twc)
        Rcw = Tcw[0:3, 0:3]
        tcw = Tcw[0:3, 3]
        semantic_obs = observation["semantic_sensor_1st_person"]
        depth_obs = observation["depth_sensor_1st_person"]

        obj_ids = np.unique(semantic_obs)
        obj_ids = np.intersect1d(obj_ids, self.ids)

        for obj_id in obj_ids:
            mask = (semantic_obs == obj_id)
            if mask is not None:
                obj_coords = np.nonzero(mask)
                z = depth_obs[obj_coords]
                obj_coords = np.asarray(obj_coords).T

                ux = obj_coords[:, 1]
                uy = obj_coords[:, 0]

                x = (ux - self.cx) * z / self.fx
                y = (uy - self.cy) * z / self.fx

                x_mean = np.mean(x)
                y_mean = np.mean(y)
                z_mean = np.mean(z)

                Oc = [x_mean, y_mean, z_mean]

                obj_range = np.sqrt(Oc[0] * Oc[0] + Oc[2] * Oc[2])
                bearing = np.arctan2(-Oc[0], Oc[2])
                range_bearing = RangeBearing()
                range_bearing.range = obj_range #+ np.random.normal(0, range_sigma)
                range_bearing.bearing = bearing #+ np.random.normal(0, bearing_sigma)
                range_bearing.id = obj_id
                range_bearing.obj_class = self.id2category[obj_id]
                alpha = np.ones(self.nClass)
                alpha[self.categories.index(self.id2category[obj_id])] = self.alpha_constant
                range_bearing.probability = dirichlet.rvs(alpha=alpha,
                                                          size=1)[0]. \
                    tolist()
                range_bearings.append(range_bearing)


        # for i in range(len(self.objs)):
        #     obj = self.objs[i]
        #     Or = np.matmul(Rrw, obj.translation) + trw
        #
        #     # Trc = np.identity(4)
        #     # Trc[0:3, 0:3] = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        #     # Trc[0:3, 3] = np.asarray([0, 0.6, 0])
        #     #
        #     # Tcr = inv(Trc)
        #     # Rcr = Tcr[0:3, 0:3]
        #     # tcr = Tcr[0:3, 3]
        #     # Oc1 = np.matmul(Rcr, Or) + tcr
        #     #
        #     # print('Object ', i, ' Oc is ', Oc1)
        #
        #
        #     Oc = np.matmul(Rcw, obj.translation) + tcw
        #     # print('Object ', i, ' Oc is ', Oc)
        #     if Oc[2] < 0:
        #         continue
        #
        #     u = np.matmul(self.K, Oc)
        #     ux = u[0]/u[2]
        #     uy = u[1]/u[2]
        #
        #     if ux < 0 or uy < 0 or (ux >= self.sim_settings["width"] - 1) or (uy >= self.sim_settings["height"] - 1):
        #         continue
        #
        #     if semantic_obs[int(uy), int(ux)]:
        #         continue
        #
        #     z = depth_obs[int(uy), int(ux)]
        #     x = (ux - self.cx)*z/self.fx
        #     y = (uy - self.cy)*z/self.fx
        #
        #     dist = norm(Oc - [x, y, z])
        #
        #     self.log_info.append(dist)
        #
        #     if dist > 0.1:
        #         continue
        #
        #     Oc = np.asarray([x, y ,z])
        #     # rgb_img = cv2.circle(rgb_img, (int(ux), int(uy)), int(5), (75, 75, 75), 2)
        #
        #     # print('Object ', i, ' ux is ', ux, ' uy is ', uy)
        #
        #     bearing = np.arctan2(-Oc[0], Oc[2])
        #     range_bearing = RangeBearing()
        #     range_bearing.range = obj_range# + np.random.normal(0, range_sigma)
        #     range_bearing.bearing = bearing# + np.random.normal(0, bearing_sigma)
        #     range_bearing.id = i + max(self.ids) + 1
        #     alpha = np.ones(self.nClass)
        #     alpha[len(self.categories) + i] = self.alpha_constant
        #     range_bearing.probability = dirichlet.rvs(alpha=alpha,
        #                                               size=1)[0].\
        #         tolist()
        #     range_bearings.append(range_bearing)


        range_msg.range_bearings = range_bearings

        # render observation
        # image_message = self.bridge.cv2_to_imgmsg(rgb_img, encoding="passthrough")
        # cv2.imwrite('view3.png', observation["color_sensor_1st_person"])
        image_message = self.bridge.cv2_to_imgmsg(
            observation["color_sensor_1st_person"], encoding="passthrough")
        image_message.encoding = "rgba8"
        depth_message = self.bridge.cv2_to_imgmsg(depth_obs, encoding="passthrough")
        # end_time = time()

        semantic_msg = BoundingBoxes()
        bounding_boxes = []
        detection_id = 0
        cv_image = observation["color_sensor_1st_person"].copy().astype(np.uint8)
        for obj_id in obj_ids:
            mask = (semantic_obs == obj_id)
            if mask is not None:
                category = self.id2category[obj_id]

                x_sum = np.any(mask, 0)
                y_sum = np.any(mask, 1)

                x_min = np.min(np.nonzero(x_sum))
                y_min = np.min(np.nonzero(y_sum))
                x_max = np.max(np.nonzero(x_sum))
                y_max = np.max(np.nonzero(y_sum))

                bounding_box = BoundingBox()
                Class_distribution = [0.0] * len(self.categories)

                bounding_box.probability = 1
                bounding_box.xmin = x_min
                bounding_box.ymin = y_min
                bounding_box.xmax = x_max
                bounding_box.ymax = y_max
                bounding_box.Class = category
                bounding_box.id = detection_id
                bounding_box.object_id = obj_id
                detection_id += 1

                Class_distribution[self.categories.index(category)] = 1.0
                bounding_box.Class_distribution = Class_distribution
                bounding_boxes.append(bounding_box)

        #         if (bounding_box.xmax - bounding_box.xmin > 25) and (
        #             bounding_box.ymax - bounding_box.ymin > 25):
        #             cv2.rectangle(cv_image, (bounding_box.xmin,
        #                                      bounding_box.ymin),
        #                           (bounding_box.xmax, bounding_box.ymax),
        #                           (0, 255, 0), 2)
        #
        #         print('category is', category)
        #         print('shape is', x_min, x_max, y_min, y_max)
        # print('added box pixel value is ', semantic_obs[402, 699], semantic_obs[401, 698])
        cv2.imwrite('virtual_box1.png', cv_image)
        cv2.imwrite('virtual_semantic.png', semantic_obs.astype(np.uint8))
        mdic = {"depth": depth_obs}
        savemat('depth.mat', mdic)
        # cv2.imwrite('depth.png', depth_obs)
        # cv2.imshow("Show", cv_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        self.camera_info_message.header.seq = self.seq
        self.camera_info_message.header.stamp = rospy.get_rostime()
        image_message.header = self.camera_info_message.header
        depth_message.header = self.camera_info_message.header

        semantic_msg.bounding_boxes = bounding_boxes
        semantic_msg.header = image_message.header
        semantic_msg.image_header = image_message.header
        range_msg.header = copy.deepcopy(image_message.header)
        range_msg.header.frame_id = 'camera_link'

        self.image_pub.publish(image_message)
        self.depth_pub.publish(depth_message)
        self.camera_info_pub.publish(self.camera_info_message)
        self.semantic_pub.publish(semantic_msg)
        self.range_pub.publish(range_msg)
        self.seq += 1

    def save_data(self):

        self.out.release()

        metric = self.semantic_map.get_metric()
        map_with_target_objects = maps.colorize_topdown_map(metric["map"])

        map_with_target_objects = draw_target_objects(map_with_target_objects, self.target_objs, self.sim)

        map_with_target_objects = maps.draw_agent(
            image=map_with_target_objects,
            agent_center_coord=metric["agent_map_coord"],
            agent_rotation=metric["agent_angle"],
            agent_radius_px=min(map_with_target_objects.shape[0:2]) // 32,
        )

        cv2.imwrite('path2_v2.png', map_with_target_objects)


        print('total path length is ', self.total_path_length)
        with open("/home/bear/Github/habitat-lab/log.pkl", "wb") as fp:  # Pickling
            pickle.dump(self.log_info, fp)

        video_prefix = "multi_2"
        if self.make_video:
            overlay_dims = (
                int(self.sim_settings["width"] / 5),
                int(self.sim_settings["height"] / 5))
            print("overlay_dims = " + str(overlay_dims))
            overlay_settings = [
                {
                    "obs": "color_sensor_1st_person",
                    "type": "color",
                    "dims": overlay_dims,
                    "pos": (10, 10),
                    "border": 2,
                },
                # {
                #     "obs": "depth_sensor_1st_person",
                #     "type": "depth",
                #     "dims": overlay_dims,
                #     "pos": (10, 30 + overlay_dims[1]),
                #     "border": 2,
                # },
                # {
                #     "obs": "semantic_sensor_1st_person",
                #     "type": "semantic",
                #     "dims": overlay_dims,
                #     "pos": (10, 50 + overlay_dims[1] * 2),
                #     "border": 2,
                # },
            ]
            print("overlay_settings = " + str(overlay_settings))
            # overlay_settings = None
            vut.make_video(
                observations=self.observations,
                primary_obs="color_sensor_3rd_person",
                primary_obs_type="color",
                video_file=self.output_path + video_prefix,
                fps=int(1.0 / self.time_step),
                open_vid=self.show_video,
                # overlay_settings=overlay_settings,
                depth_clip=10.0,
            )

        # remove locobot while leaving the agent node for later use
        self.rigid_obj_mgr.remove_object_by_id(self.locobot_obj.object_id,
                                               delete_object_node=False)
        self.rigid_obj_mgr.remove_all_objects()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video",
                        action="store_false")
    parser.set_defaults(show_video=True, make_video=False)
    args, _ = parser.parse_known_args()

    rospy.init_node("habitat")

    env = Environment(args)
    # env.run()

    rospy.loginfo("Press Ctrl + C to terminate")
    try:
        # env.run_rotate()
        env.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
