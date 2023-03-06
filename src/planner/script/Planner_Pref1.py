#!/usr/bin/env python2
import numpy
import numpy as np
from numpy.linalg import inv, norm, eig, det
from scipy.interpolate import interp1d
import rospy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from darknet_ros_msgs.msg import Object2D, Object2DArray, RobotCommand
from nav_msgs.msg import Odometry, OccupancyGrid
from tf2_ros import TransformException, ConnectivityException, \
    ExtrapolationException
from cv_bridge import CvBridge
import cv2
import tf
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as svn
from scipy.stats import dirichlet
from skimage.draw import line
from scipy import io
from scipy.special import softmax
import math
from time import time
from threading import Thread, Lock
from datetime import datetime
import actionlib
from ipa_building_msgs.msg import MapSegmentationAction, MapSegmentationGoal
from darknet_ros_msgs.srv import *
from DFA import DFA


def color_converter(value):
    if value == -1:
        return 128
    elif value == 0:
        return 255
    else:
        return 0


class MapObject:
    def __init__(self, object_id, pos, pos_var, class_probs):
        self.id = object_id
        self.pos = pos
        self.pos_var = pos_var
        self.class_probs = class_probs
        self.visualized = False
        # self.grip_pub = rospy.Publisher("/grip_image_" + str(object_id), Image, queue_size=10)

    def update(self, pos, pos_var, class_probs):
        self.pos = pos
        self.pos_var = pos_var
        self.class_probs = class_probs


class Planner:
    def __init__(self):
        self.target_frame = "camera_link"
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.ang_var = None
        self.pos_var = None

        self.old_ang = None
        self.ang = None
        self.pos = None
        self.vel = None

        self.sigma_p = None
        self.sigma_p_inv = None

        self.seg_map = None
        self.old_seg_map = None

        self.grid_map_grey = None
        self.old_grid_map = None
        self.old_obstacle_map = None

        self.old_sighting_map = None
        self.old_sighting_phase = None
        self.old_sighting_prob = None

        self.old_frontier_map = None
        self.old_frontier_prob = None
        self.old_frontier_phase = None

        self.grid_map_color = None
        self.obstacle_map = None
        self.obstacle_prob = None

        self.goal_map = {}
        self.goal_phase = {}
        self.goal_prob = {}

        self.old_goal_map = None
        self.old_goal_prob = None
        self.old_goal_phase = None

        self.frontier_phase = None
        self.frontier_map = None
        self.frontier_prob = None
        self.frontier_centroid = None
        self.frontier_num_labels = None
        self.frontier_stat = None

        self.width = None
        self.height = None

        self.old_height = None
        self.old_width = None

        self.resolution = None
        self.old_resolution = None

        self.offset_x = None
        self.offset_y = None

        self.old_offset_x = None
        self.old_offset_y = None

        self.objects = {}
        self.sightings_map = {}
        self.sightings_phase = {}
        self.sightings_prob = {}
        self.room_ids_2_objects = {}
        self.room_ids_2_prob = {}

        self.bridge = CvBridge()

        self.seq = 1

        self.goal = None
        self.goal_status = False
        self.mission_status = False
        self.planner_type = None
        self.ang_goal1_status = False
        self.ang_goal2_status = False
        self.ang_calculation = True

        self.ang_goal1 = None
        self.ang_goal2 = None

        self.initial_obs = True
        self.replan = True
        self.observed_ang = 0

        self.blind_radius = 1.0
        self.goal_radius = 1.0
        self.action_deltas = [[1, -1],
                              [1, 0],
                              [1, 1],
                              [0, 1],
                              [-1, 1],
                              [-1, 0],
                              [-1, -1],
                              [0, -1]]
        self.nA = 8

        self.action_probabilities = (0.8, 0.1, 0.1)

        self.V = {}
        self.gamma = 0.99
        self.P = {}
        self.states_index = None
        self.policy = None

        self.classes = ['towel', 'objects', 'lighting', 'stool', 'counter',
                        'door', 'clothes', 'appliances', 'furniture',
                        'shelving', 'bed', 'blinds', 'table', 'cabinet',
                        'shower', 'chair', 'chest_of_drawers', 'tv_monitor',
                        'toilet', 'mirror', 'sofa', 'cushion', 'sink',
                        'banana', 'cheezit']
        self.target_class = 'towel'
        self.target_obj_id = [92, 85]

        self.evidence_classes = ['toilet', 'lighting', 'sink', 'mirror',
                                 'shower']
        self.obj_of_interest = None
        self.old_obj_of_interest = None
        self.max_class_prob = 0
        self.old_max_class_prob = 0

        self.agent_radius = 0.15 #0.15
        self.angular_error_threshold = 0.5
        self.max_linear_speed = 0.2
        self.max_turn_speed = 0.4
        self.planning_time = 0
        self.start_time = 0
        self.end_time = 0

        self.dfa = None
        self.construct_dfa()
        self.dfa_state = self.dfa.initial_state

        self.mutex = Lock()
        self.value_mutex = Lock()

        self.map_client = actionlib.SimpleActionClient(
            '/room_segmentation/room_segmentation_server',
            MapSegmentationAction)
        rospy.loginfo("Waiting for action server to start.")
        # wait for the action server to start
        self.map_client.wait_for_server()  # will wait for infinite time
        rospy.loginfo("Action server started, sending goal.")

        self.odom_sub = rospy.Subscriber("/rtabmap/odom", Odometry,
                                         self.odom_callback)
        self.semantic_map_sub = rospy.Subscriber("/semantic_map",
                                                 Object2DArray,
                                                 self.semantic_map_callback)
        self.map_sub = rospy.Subscriber("/rtabmap/grid_map", OccupancyGrid,
                                        self.map_callback)

        self.map_click_sub = rospy.Subscriber("/map_image_mouse_left", Point,
                                              self.map_click_callback)
        self.image_pub = rospy.Publisher("/map_image", Image, queue_size=10)
        self.debug_pub = rospy.Publisher("/debug_image", Image, queue_size=10)
        self.object_info_pub = rospy.Publisher("/obj_info", Image,
                                               queue_size=10)
        self.command_pub = rospy.Publisher("/robot_command", RobotCommand,
                                           queue_size=10)
        self.obstacle_map_pub = rospy.Publisher("/obstacle_map", Image,
                                                queue_size=10)
        self.frontier_map_pub = rospy.Publisher("/frontier_map", Image,
                                                queue_size=10)
        self.seg_map_pub = rospy.Publisher("/seg_map", Image, queue_size=10)
        self.sightings_map_pub = rospy.Publisher("/sighting_map",
                                                 Image, queue_size=10)
        self.goal_map_pub = rospy.Publisher("/goal_map", Image, queue_size=10)
        self.mission_status_pub = rospy.Publisher("/mission_status",
                                                  Image, queue_size=10)
        self.planning_time_pub = rospy.Publisher("/planning_time",
                                                  Float64, queue_size=10)
        self.total_time_pub = rospy.Publisher("/total_time",
                                                Float64, queue_size=10)
        self.c = rospy.ServiceProxy('BBN_infer', BBNInfer)
        rospy.wait_for_service('BBN_infer')
        self.rate = rospy.Rate(20)  # 3hz
        self.dt = 1 / 30.0
        self.iteration = 0

        rospy.on_shutdown(self.save_data)

    def construct_dfa(self):
        self.dfa = DFA(2, ['1','2', '3','4', '5', '6', '7', '8', '9','10', '11', '12', '13', '14', '15', '16', 'D'], final_states={0, 5, 4}, goal_states={5:0.1, 0:1})
        # '1' !A & C & !E & !D
        # '2' A & C & !E & !D
        # '3' A & !C & !E & !D
        # '4' !A & !C & !E & !D

        # '5' !A & B & !E & !D
        # '6' A & B & !E & !D
        # '7' A & !B & !E & !D
        # '8' !A & !B & !E & !D

        # '9' !A & !C & E & !D
        # '13' !A & C & E & !D
        # '15' A & C & E & !D

        # '10' A & !B & E & !D
        # '14' A & B & E & !D
        # '16' !A & B & E & !D

        # '11' A & !C & E & !D
        # '12' !A & E & !B & !D

        # D hit obstacle

        self.dfa.add_transition('1', 2, 0)
        self.dfa.add_transition('6', 2, 0)
        self.dfa.add_transition('13', 2, 0)
        self.dfa.add_transition('14', 2, 0)

        self.dfa.add_transition('4', 2, 1)
        self.dfa.add_transition('7', 2, 3)

        self.dfa.add_transition('9', 2, 5)
        self.dfa.add_transition('10', 2, 5)

        self.dfa.add_transition('1', 1, 0)
        self.dfa.add_transition('2', 1, 0)
        self.dfa.add_transition('13', 1, 0)
        self.dfa.add_transition('15', 1, 0)

        self.dfa.add_transition('3', 1, 1)
        self.dfa.add_transition('4', 1, 1)

        self.dfa.add_transition('11', 1, 5)
        self.dfa.add_transition('9', 1, 5)

        self.dfa.add_transition('5', 3, 0)
        self.dfa.add_transition('6', 3, 0)
        self.dfa.add_transition('14', 3, 0)
        self.dfa.add_transition('16', 3, 0)

        self.dfa.add_transition('7', 3, 3)
        self.dfa.add_transition('8', 3, 3)

        self.dfa.add_transition('12', 3, 5)
        self.dfa.add_transition('10', 3, 5)

        self.dfa.add_transition('D', 1, 4)
        self.dfa.add_transition('D', 2, 4)
        self.dfa.add_transition('D', 3, 4)

    def label_function_prob(self, state):
        # D hit obstacle
        labels = {}

        D = self.old_obstacle_prob[state[0], state[1]]
        B = self.old_goal_prob[self.old_obj_of_interest][state[0], state[1]]*self.old_max_class_prob
        E = self.old_frontier_prob[state[0], state[1]]
        C = self.old_sighting_prob[self.old_obj_of_interest][state[0], state[1]]*self.old_max_class_prob

        labels['D'] = D

        if self.old_max_class_prob > 0.9:
            # '2' A & C & !E & !D
            # '3' A & !C & !E & !D
            # '6' A & B & !E & !D
            # '7' A & !B & !E & !D

            # '15' A & C & E & !D
            # '10' A & !B & E & !D
            # '14' A & B & E & !D
            # '11' A & !C & E & !D

            labels['2'] = C*(1-E)*(1-D)
            labels['3'] = (1-C)*(1-E)*(1-D)
            labels['6'] =  B*(1-E)*(1-D)
            labels['7'] = (1-B)*(1-E)*(1-D)

            labels['15'] = C*E*(1-D)
            labels['10'] = (1-B)*E*(1-D)
            labels['14'] = B*E*(1-D)
            labels['11'] = E * (1 - C) * (1 - D)

        else:
            # '1' !A & C & !E & !D
            # '4' !A & !C & !E & !D
            # '5' !A & B & !E & !D
            # '8' !A & !B & !E & !D

            # '9' !A & !C & E & !D
            # '13' !A & C & E & !D
            # '16' !A & B & E & !D
            # '12' !A & E & !B & !D

            labels['1'] = C * (1 - E) * (1 - D)
            labels['4'] = (1 - C) * (1 - E) * (1 - D)
            labels['5'] = B * (1 - E) * (1 - D)
            labels['8'] = (1 - B) * (1 - E) * (1 - D)

            labels['9'] = (1 - C) * E * (1 - D)
            labels['13'] = C * E * (1 - D)
            labels['16'] = B * E * (1 - D)
            labels['12'] = E * (1 - B) * (1 - D)

        return labels

    def label_function(self, pos_grid):
        symbols = []

        # D hit obstacle
        if self.old_obstacle_map[pos_grid[1], pos_grid[0]]:
            symbols.append('D')
        else:
            if self.old_max_class_prob > 0.9:
                if self.old_frontier_map[pos_grid[1], pos_grid[0]]:
                    # '10' A & !B & E & !D
                    # '14' A & B & E & !D
                    if self.old_goal_map[self.old_obj_of_interest][pos_grid[1], pos_grid[0]]:
                        symbols.append('14')
                    else:
                        symbols.append('10')

                    # '15' A & C & E & !D
                    # '11' A & !C & E & !D
                    if self.old_sighting_map[self.old_obj_of_interest][pos_grid[1], pos_grid[0]]:
                        symbols.append('15')
                    else:
                        symbols.append('11')

                else:
                    # '6' A & B & !E & !D
                    # '7' A & !B & !E & !D
                    if self.old_goal_map[self.old_obj_of_interest][pos_grid[1], pos_grid[0]]:
                        symbols.append('6')
                    else:
                        symbols.append('7')

                    # '2' A & C & !E & !D
                    # '3' A & !C & !E & !D
                    if self.old_sighting_map[self.old_obj_of_interest][pos_grid[1], pos_grid[0]]:
                        symbols.append('2')
                    else:
                        symbols.append('3')
            else:
                if self.old_frontier_map[pos_grid[1], pos_grid[0]]:
                    # '16' !A & B & E & !D
                    # '12' !A & !B & E & !D
                    if self.old_goal_map[self.old_obj_of_interest][pos_grid[1], pos_grid[0]]:
                        symbols.append('16')
                    else:
                        symbols.append('12')

                    # '13' !A & C & E & !D
                    # '9' !A & !C & E & !D
                    if self.old_sighting_map[self.old_obj_of_interest][pos_grid[1], pos_grid[0]]:
                        symbols.append('13')
                    else:
                        symbols.append('9')

                else:
                    # '5' !A & B & !E & !D
                    # '8' !A & !B & !E & !D
                    if self.old_goal_map[self.old_obj_of_interest][pos_grid[1], pos_grid[0]]:
                        symbols.append('5')
                    else:
                        symbols.append('8')

                    # '1' !A & C & !E & !D
                    # '4' !A & !C & !E & !D
                    if self.old_sighting_map[self.old_obj_of_interest][pos_grid[1], pos_grid[0]]:
                        symbols.append('1')
                    else:
                        symbols.append('4')
        return symbols

    def update_dfa_state(self, symbols):
        if self.dfa_state in self.dfa.final_states:
            return

        next_state = None

        for symbol in symbols:
            next_state = self.dfa.get_transition(symbol, self.dfa_state)
            if next_state is not None:
                self.dfa_state = next_state
                return

        assert next_state is not None

    def save_date(self):
        rospy.loginfo('total_planning time is ', self.planning_time)

    def acquire_transition(self, state):
        state_actions = self.P.get(state, False)
        dfa_state = state[1]
        if not state_actions:
            self.P[state] = {}
            for action in range(self.nA):
                action_deltas = np.take(self.action_deltas, [action, action + 1, action - 1], axis=0, mode='wrap')
                next_states = np.unravel_index(state[0], (self.old_height, self.old_width)) + action_deltas

                next_states[next_states[:, 0] < 0, 0] = 0
                next_states[next_states[:, 1] < 0, 1] = 0
                next_states[next_states[:, 0] > self.old_height - 1, 0] = self.old_height - 1
                next_states[next_states[:, 1] > self.old_width - 1, 1] = self.old_width - 1

                next_states_indices = np.ravel_multi_index(next_states.T, (self.old_height, self.old_width))

                out_of_free_grids = np.in1d(next_states_indices, self.states_index, invert=True)
                pnsrt = []

                for i in range(3):
                    next_state = next_states[i]
                    labels = self.label_function_prob(next_state)
                    for label in labels:
                        if labels[label] > 0:
                            next_dfa_state = None
                            if dfa_state in self.dfa.final_states:
                                next_dfa_state = dfa_state
                            else:
                                next_dfa_state = self.dfa.get_transition(label, self.dfa_state)
                            terminal = out_of_free_grids[i] or (next_dfa_state in self.dfa.final_states)
                            reward = self.dfa.goal_states.get(next_dfa_state, 0)
                            if next_dfa_state is not None:
                                pnsrt.append((labels[label]*self.action_probabilities[i], (next_states_indices[i], next_dfa_state), reward, terminal))

                self.P[state][action] = pnsrt
            return self.P.get(state, False)
        else:
            return state_actions

    def select_action(self, state, train=False):
        Q_s = np.zeros(self.nA)
        for a in range(self.nA):
            StateActions = self.acquire_transition(state)
            if not StateActions:
                return -1
            if len(StateActions) != 8:
                print(StateActions)

            transitions = StateActions[a]
            for transition in transitions:
                probability = transition[0]
                nextstate = transition[1]
                reward = transition[2]
                terminal = transition[3]

                if terminal:
                    Q_s[a] = Q_s[a] + probability * reward
                else:
                    Q_s[a] = Q_s[a] + probability * (reward + self.gamma * self.V[nextstate])

        # select an action
        if train:
            action_probs = softmax(Q_s)
            action = np.random.choice(self.nA, p=action_probs)
            self.value_mutex.acquire()
            self.V[state] = np.max(Q_s)
            self.value_mutex.release()
        else:
            actions = np.flatnonzero(Q_s == np.max(Q_s))
            action = np.random.choice(actions)
        return action

    def trial(self, s0):
        done = False
        state = (s0, self.dfa_state)
        states = []
        while not done:
            action = self.select_action(state, train=True)
            # return a reward and new state
            transitions = self.P[state][action]
            probabilities, nextstates, rewards, terminals = zip(*transitions)

            if np.min(probabilities) < 0:
                probabilities = np.asarray(probabilities) - np.min(
                    probabilities)
            probabilities = probabilities / np.sum(probabilities)

            index = np.random.choice(len(nextstates), p=probabilities)

            new_state = nextstates[index]
            done = terminals[index]

            # append state, action, reward to episode
            states.append(state)
            # update state to new state
            state = new_state

            if len(states) > 10 * (self.old_width + self.old_height):
                break

    def run(self):
        while not rospy.is_shutdown():
            if self.obstacle_map is not None and not self.initial_obs and not self.mission_status:
                pos_grid = np.asarray(self.to_grid(self.pos))
                # if self.grid_map_grey[pos_grid[1], pos_grid[0]] != 255:
                #     print('replan because hit obstacles')
                #     self.replan = True
                #     continue
                start_time = time()
                if self.replan:
                    print('start replanning')
                    self.iteration = 0
                    self.mutex.acquire()
                    print('acquire mutex')
                    self.old_grid_map = self.grid_map_grey.copy()
                    #self.old_seg_map = self.seg_map.copy()

                    self.old_obstacle_map = self.obstacle_map.copy()
                    self.old_obstacle_prob = self.obstacle_prob.copy()

                    self.old_sighting_map = self.sightings_map.copy()
                    self.old_sighting_prob = self.sightings_prob.copy()
                    self.old_sighting_phase = self.sightings_phase.copy()

                    self.old_frontier_map = self.frontier_map.copy()
                    self.old_frontier_prob = self.frontier_prob.copy()
                    self.old_frontier_phase = self.frontier_phase.copy()

                    self.old_goal_map = self.goal_map.copy()
                    self.old_goal_prob = self.goal_prob.copy()
                    self.old_goal_phase = self.goal_phase.copy()

                    self.old_height = self.height
                    self.old_width = self.width
                    self.old_resolution = self.resolution
                    self.old_offset_x = self.offset_x
                    self.old_offset_y = self.offset_y

                    self.old_obj_of_interest = self.obj_of_interest
                    self.old_max_class_prob = self.max_class_prob
                    print('copy data')
                    self.mutex.release()
                    print('mutex release')
                    self.planner_type = 'Preference'

                    self.dfa_state = self.dfa.initial_state

                    state_space_grid = (self.old_grid_map == 255).astype(np.uint8)
                    states = np.nonzero(state_space_grid)

                    # kernel = np.ones((3, 3), np.uint8)
                    # state_space_grid = cv2.dilate(free_space_grid, kernel, iterations=1, borderType=cv2.BORDER_ISOLATED)

                    self.states_index = np.ravel_multi_index(states, (self.old_height, self.old_width))
                    states = np.asarray(states).T

                    self.V = {}
                    self.P = {}
                    kernel = np.ones((3, 3), np.uint8)
                    Vh = None
                    if self.max_class_prob> 0.9:
                        Vh = self.dfa.goal_states[0]*self.old_goal_map[self.old_obj_of_interest] + \
                             self.dfa.goal_states[5]*self.old_frontier_map
                    else:
                        Vh = self.dfa.goal_states[0] * self.old_sighting_map[
                            self.old_obj_of_interest] +\
                             self.dfa.goal_states[5] * self.old_frontier_map
                    Vh = np.minimum(Vh, 1)
                    count = 0
                    non_states = (self.old_grid_map != 255)
                    while Vh[pos_grid[1], pos_grid[0]] == 0:
                        Vh_new = cv2.dilate(Vh, kernel, iterations=1,
                                            borderType=cv2.BORDER_ISOLATED) * self.gamma
                        Vh = np.maximum(Vh, Vh_new)
                        Vh[non_states] = 0
                        count += 1

                    Vh[Vh == 0] = np.amax(Vh)*np.power(self.gamma, count)

                    non_final_dfa_states = set(self.dfa.states).difference(self.dfa.final_states)
                    for dfa_state in non_final_dfa_states:
                        for j in range(states.shape[0]):
                            state_index = self.states_index[j]
                            state = states[j]
                            self.V[(state_index, dfa_state)] = Vh[state[0], state[1]]
                    print('construct heuristic')
                    self.replan = False
                start_time = time()

                pos_index = np.ravel_multi_index(np.flipud(pos_grid), (self.old_height, self.old_width))

                if pos_index not in self.states_index:
                    print('replan because enter unknown states')
                    self.replan = True
                    continue

                symbols = self.label_function(pos_grid)
                self.update_dfa_state(symbols)
                self.goal_status = self.dfa_state in self.dfa.final_states
                self.trial(pos_index)
                self.iteration += 1

                # self.policy, self.V = value_iteration(P, self.nA)
                value_image = np.zeros((self.old_height, self.old_width))
                for product_mdp_state in self.V:
                    state_index = product_mdp_state[0]
                    if state_index > (self.old_width * self.old_height - 1):
                        continue
                    state = np.unravel_index(state_index,
                                             (self.old_height, self.old_width))
                    value_image[state[0], state[1]] = self.V[product_mdp_state]

                # value_image = 255/np.amax(value_image)*value_image
                # value_image = np.clip(value_image, 0, 255).astype(np.uint8)
                debug_image_color = cv2.cvtColor(self.old_grid_map,
                                                 cv2.COLOR_GRAY2BGR)

                # path = set()
                # while True:
                #     debug_image_color[pos_grid[1], pos_grid[0], :] = [189, 114,
                #                                                       0]
                #     if tuple(pos_grid) in path:
                #         break
                #     else:
                #         path.add(tuple(pos_grid))
                #     pos_index = np.ravel_multi_index(np.flipud(pos_grid),
                #                                      (self.old_height,
                #                                       self.old_width))
                #     # if self.policy.get(pos_index, -1) < 0:
                #     #     print('this grid has no policy')
                #     #     break
                #     # else:
                #     # action_index = self.policy[pos_index]
                #     action_index = self.select_action(pos_index)
                #     pos_grid = pos_grid + np.flipud(
                #         self.action_deltas[action_index])

                # for state_index in states:
                #     state = np.unravel_index(state_index,
                #                              (self.old_height, self.old_width))
                #     debug_image_color[state[0], state[1], :] = [189, 114, 0]

                image_message = self.bridge.cv2_to_imgmsg(debug_image_color,
                                                          encoding="bgr8")
                self.debug_pub.publish(image_message)
                end_time = time()

                #print('time spent in iteration is ', end_time - start_time)
                font = cv2.FONT_HERSHEY_PLAIN
                font_scale = 2  # 0.6
                font_thickness = 1
                mission_info_image = np.zeros((250, 600), dtype=np.uint8)
                text_coord = [0, 0]
                text1 = "DFA state: " + str(self.dfa_state)
                text2 = "Intermediate goal reached: " + str(self.goal_status)
                text3 = "Most probable object: " + str("%.2f" % self.max_class_prob)
                text4 = "Current planner: " + self.planner_type
                text5 = "Value Iteration num: " + str(self.iteration)
                texts = [text1, text2, text3, text4, text5]
                for text in texts:
                    text_size, _ = cv2.getTextSize(text, font, font_scale,
                                                   font_thickness)
                    _, text_h = text_size
                    text_coord[1] = text_coord[1] + text_h + 25
                    cv2.putText(mission_info_image, text, tuple(text_coord),
                                font,
                                font_scale, 255, font_thickness)

                image_message = self.bridge.cv2_to_imgmsg(mission_info_image,
                                                          encoding="mono8")
                self.mission_status_pub.publish(image_message)
                end_time = time()
                if self.iteration < 300:
                    self.planning_time += (end_time - start_time)
                if self.iteration == 300:
                    self.planning_time_pub.publish(self.planning_time)
                self.total_time_pub.publish(end_time - self.start_time)
            # self.rate.sleep()

    def angle_tracking(self, angle_error):
        rot_dir = 1.0
        if angle_error < 0:
            rot_dir = -1.0

        angular_correction = 0.0
        if np.abs(angle_error) > (
            self.max_turn_speed * 10.0 * self.dt):
            angular_correction = self.max_turn_speed
        else:
            angular_correction = np.abs(angle_error) / 2.0

        angular_correction = np.clip(rot_dir * angular_correction,
                                     -self.max_turn_speed,
                                     self.max_turn_speed)

        return angular_correction

    def odom_callback(self, msg):
        # only takes the covariance, the pose is taken from tf transformation
        self.pos_var = msg.pose.covariance[0]
        self.ang_var = msg.pose.covariance[-1]

        self.sigma_p = np.diag([self.pos_var, self.pos_var, self.ang_var])
        self.sigma_p_inv = inv(self.sigma_p)

        self.vel = msg.twist.twist.linear.x

        from_frame_rel = self.target_frame
        to_frame_rel = 'map'

        try:
            trans = self.tfBuffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rospy.Time(0))

            self.pos = np.asarray([trans.transform.translation.x,
                                   trans.transform.translation.y])

            self.old_ang = self.ang
            self.ang = tf.transformations.euler_from_quaternion(
                [trans.transform.rotation.x,
                 trans.transform.rotation.y,
                 trans.transform.rotation.z,
                 trans.transform.rotation.w])[-1]

        except (TransformException, ConnectivityException,
                ExtrapolationException) as ex:
            rospy.loginfo('Could not transform %s to %s: ', to_frame_rel,
                          from_frame_rel)
            return

        if self.initial_obs and self.old_ang is not None:
            self.observed_ang += np.arctan2(
                np.sin(self.ang - self.old_ang),
                np.cos(self.ang - self.old_ang))

            if self.observed_ang > 2 * np.pi:
                self.initial_obs = False
                self.start_time = time()

        command_msg = RobotCommand()
        if self.initial_obs:
            command_msg.linear_vel = 0
            command_msg.angular_vel = self.max_turn_speed
        elif self.mission_status:
            command_msg.linear_vel = 0
            command_msg.angular_vel = 0
        else:
            pos_grid = self.to_grid(self.pos)
            if self.grid_map_grey[pos_grid[1], pos_grid[0]] != 255:
                grid_map_convervative = cv2.erode(self.grid_map_grey, np.ones((3,3), dtype=np.uint8), iterations=1)

                free_pnts = np.asarray(np.where(grid_map_convervative == 255)).T
                distances = norm(free_pnts - np.flipud(pos_grid), axis=1)
                goal_grid = free_pnts[np.argmin(distances), :]

                self.goal = self.to_map(np.flipud(goal_grid))
                to_waypoint = self.goal - self.pos
                angle_error = np.arctan2(to_waypoint[1],
                                         to_waypoint[0]) - self.ang
                angle_error = np.arctan2(np.sin(angle_error),
                                         np.cos(angle_error))

                if np.abs(angle_error) < self.angular_error_threshold:
                    # speed up to max
                    # the positive direction of car bearing is opposite in
                    # simulation
                    new_velocity = (-self.vel - self.max_linear_speed) / 2.0
                else:
                    # slow down to 0
                    new_velocity = -self.vel / 2.0

                command_msg.linear_vel = new_velocity
                command_msg.angular_vel = self.angle_tracking(angle_error)

            elif self.iteration > 300 and not self.goal_status and not self.replan:
                pos_grid = self.to_grid_old(self.pos)
                pos_index = np.ravel_multi_index(np.flipud(pos_grid),
                                                 (self.old_height,
                                                  self.old_width))
                symbols = self.label_function(pos_grid)
                self.update_dfa_state(symbols)
                # action_index = self.policy[pos_index]
                if self.old_grid_map[pos_grid[1], pos_grid[0]] == 255:
                    self.value_mutex.acquire()
                    action_index = self.select_action((pos_index, self.dfa_state))
                    self.value_mutex.release()

                    if action_index >= 0:
                        goal_grid = np.flipud(
                            self.action_deltas[action_index]) + \
                                    np.asarray(pos_grid)

                        # if self.goal is None:
                        self.goal = self.to_map_old(goal_grid)

                # goal = np.asarray([0, 1])

                to_waypoint = self.goal - self.pos
                angle_error = np.arctan2(to_waypoint[1],
                                         to_waypoint[0]) - self.ang
                angle_error = np.arctan2(np.sin(angle_error),
                                         np.cos(angle_error))

                # print('current grid is ', pos_grid)
                # print('goal grid is ', goal_grid)
                # print('current pose is ', self.pos)
                # print('goal pose is ', self.goal)
                # print('actual angle is ', self.ang)
                # print('desired angle is ', np.arctan2(to_waypoint[1], to_waypoint[0]))
                #
                # print('angular error is ', angle_error)

                # print('error is', np.linalg.norm(to_waypoint), angle_error)

                if np.abs(angle_error) < self.angular_error_threshold:
                    # speed up to max
                    # the positive direction of car bearing is opposite in
                    # simulation
                    new_velocity = (-self.vel - self.max_linear_speed) / 2.0
                else:
                    # slow down to 0
                    new_velocity = -self.vel / 2.0

                command_msg.linear_vel = new_velocity
                command_msg.angular_vel = self.angle_tracking(angle_error)
            else:
                if self.goal_status:
                    goal_phase = None
                    if self.dfa_state == 0:
                        if self.old_max_class_prob > 0.9:
                            goal_phase = self.old_goal_phase[self.old_obj_of_interest]
                        else:
                            goal_phase = self.old_sighting_phase[self.old_obj_of_interest]
                    else:
                        goal_phase = self.old_frontier_phase


                    if self.ang_calculation:
                        pos_grid = self.to_grid_old(self.pos)
                        angle_error1 = goal_phase[pos_grid[1],
                                                       pos_grid[0]] \
                                       - self.ang + 0.3 * np.pi

                        angle_error2 = goal_phase[pos_grid[1],
                                                       pos_grid[0]] \
                                       - self.ang - 0.3 * np.pi

                        angle_error1 = np.arctan2(np.sin(angle_error1),
                                                  np.cos(angle_error1))

                        angle_error2 = np.arctan2(np.sin(angle_error2),
                                                  np.cos(angle_error2))

                        if np.abs(angle_error1) < np.abs(angle_error2):
                            self.ang_goal1 = goal_phase[pos_grid[1],
                                                             pos_grid[0]] \
                                             + 0.3 * np.pi

                            self.ang_goal2 = goal_phase[pos_grid[1],
                                                             pos_grid[0]] \
                                             - 0.3 * np.pi
                        else:
                            self.ang_goal1 = goal_phase[pos_grid[1],
                                                             pos_grid[0]] \
                                             - 0.3 * np.pi

                            self.ang_goal2 = goal_phase[pos_grid[1],
                                                             pos_grid[0]] \
                                             + 0.3 * np.pi

                    self.ang_calculation = False

                    if not self.ang_goal1_status:
                        angle_error = self.ang_goal1 - self.ang
                        angle_error = np.arctan2(np.sin(angle_error),
                                                 np.cos(angle_error))

                        command_msg.linear_vel = 0
                        command_msg.angular_vel = self.angle_tracking(
                            angle_error)

                        if np.abs(angle_error) < 0.1:
                            self.ang_goal1_status = True

                        #print('tracking angular goal 1')
                    elif not self.ang_goal2_status:
                        angle_error = self.ang_goal2 - self.ang
                        angle_error = np.arctan2(np.sin(angle_error),
                                                 np.cos(angle_error))

                        command_msg.linear_vel = 0
                        command_msg.angular_vel = self.angle_tracking(
                            angle_error)

                        if np.abs(angle_error) < 0.1:
                            self.ang_goal2_status = True

                        #print('tracking angular goal 2')
                    else:
                        self.replan = True
                        self.goal_status = False

                        self.ang_goal1_status = False
                        self.ang_goal2_status = False
                        self.ang_calculation = True

                        command_msg.linear_vel = 0
                        command_msg.angular_vel = 0

        self.command_pub.publish(command_msg)

    def to_grid(self, pose):
        return (int(round((pose[0] - self.offset_x) / self.resolution)),
                self.height - int(
                    round((pose[1] - self.offset_y) / self.resolution)))

    def to_map(self, coord):
        return (self.offset_x + self.resolution * coord[0],
                self.offset_y + (self.height - coord[1]) * self.resolution)

    def to_grid_old(self, pose):
        return (
        int(round((pose[0] - self.old_offset_x) / self.old_resolution)),
        self.old_height - int(
            round((pose[1] - self.old_offset_y) / self.old_resolution)))

    def to_map_old(self, coord):
        return (self.old_offset_x + self.old_resolution * coord[0],
                self.old_offset_y + (
                        self.old_height - coord[1]) * self.old_resolution)

    def save_data(self):
        pass

    def map_click_callback(self, msg):
        if not self.objects:
            return
        clicked_pt = np.asarray([msg.x, msg.y])
        dists = []
        obj_ids = []
        for obj_id, obj in self.objects.items():
            object_grid = np.asarray(self.to_grid(obj.pos))
            dist = norm(clicked_pt - object_grid)
            obj_ids.append(obj_id)
            dists.append(dist)

        clicked_object = self.objects[obj_ids[np.argmin(dists)]]
        if clicked_object.visualized:
            clicked_object.visualized = False
            print('now is false')
        else:
            clicked_object.visualized = True
            print('now is true')

    def map_callback(self, msg):
        """Callback from map
        :type msg: OccupancyGrid
        """
        np_data = np.array([color_converter(e) for e in msg.data])
        room_data = np_data.copy().astype(np.uint8)
        room_data[room_data == 128] = 0
        room_data = np.flipud(np.reshape(room_data, (msg.info.height,
                                                     msg.info.width)))
        goal = MapSegmentationGoal()
        goal.map_origin.position.x = msg.info.origin.position.x
        goal.map_origin.position.y = msg.info.origin.position.y
        goal.map_resolution = msg.info.resolution
        goal.return_format_in_meter = False
        goal.return_format_in_pixel = True
        goal.robot_radius = self.agent_radius

        if self.mutex.acquire(False):
            self.resolution = msg.info.resolution

            self.height = msg.info.height
            self.width = msg.info.width

            self.offset_x = msg.info.origin.position.x
            self.offset_y = msg.info.origin.position.y

            self.grid_map_grey = np.flipud(np.reshape(np_data,
                                                      (self.height, self.width)
                                                      )).astype('uint8')

            if self.pos is not None:
                roi = np.zeros_like(self.grid_map_grey)
                cv2.circle(roi, self.to_grid(self.pos),
                           int(self.blind_radius / self.resolution), 1, -1)

                mask = np.logical_and(self.grid_map_grey == 128,
                                      roi.astype(bool))

                self.grid_map_grey[mask] = 255

                room_data[mask] = 255
                goal.input_map = self.bridge.cv2_to_imgmsg(room_data,
                                                           encoding="mono8")
                # self.debug_pub.publish(goal.input_map)
                self.map_client.send_goal(goal)
                self.grid_map_color = cv2.cvtColor(self.grid_map_grey,
                                                   cv2.COLOR_GRAY2BGR)
                if self.objects:
                    self.find_sightings()
                    #print('completed find sighting')

                    self.find_goal_region()

                self.findFrontier()

                sigma_x = np.sqrt(self.pos_var) / self.resolution
                kx = int(math.ceil(
                    max(3 * np.sqrt(self.pos_var) / self.resolution, 1)))
                kx = 2 * kx + 1

                self.obstacle_map = (np.not_equal(self.grid_map_grey, 255)
                                     ).astype(float)

                self.obstacle_prob = cv2.GaussianBlur(self.obstacle_map,
                                                     (kx, kx), sigma_x,
                                                     borderType=cv2.BORDER_ISOLATED)

                self.frontier_prob = cv2.GaussianBlur(
                    (self.frontier_map > 0).astype(float),
                    (kx, kx), sigma_x,
                    borderType=cv2.BORDER_ISOLATED)

                for obj_id in self.sightings_map:
                    self.sightings_prob[obj_id] = \
                        cv2.GaussianBlur(self.sightings_map[obj_id], (kx, kx),
                                         sigma_x,
                                         borderType=cv2.BORDER_ISOLATED)

                for obj_id in self.goal_map:
                    self.goal_prob[obj_id] = \
                        cv2.GaussianBlur(self.goal_map[obj_id], (kx, kx),
                                         sigma_x,
                                         borderType=cv2.BORDER_ISOLATED)

                image_message = self.bridge.cv2_to_imgmsg(
                    (self.obstacle_map * 255).astype(np.uint8),
                    encoding="mono8")
                self.obstacle_map_pub.publish(image_message)

            # start_time = time()
            # if self.pos is not None and self.map_client.wait_for_result():
            #     result_seg = self.map_client.get_result()
            #     result_seg_img = result_seg.segmented_map
            #     self.seg_map = self.bridge.imgmsg_to_cv2(result_seg_img,
            #                                              desired_encoding="passthrough")
            #
            #     seg_map = self.seg_map.copy()
            #     seg_map[self.seg_map == 65280] = 0
            #     seg_map = seg_map * 255 / np.max(seg_map)
            #     image_message3 = self.bridge.cv2_to_imgmsg(
            #         seg_map.astype(np.uint8), encoding="mono8")
            #     self.seg_map_pub.publish(image_message3)
            #
            #     self.room_ids_2_objects = {}
            #     free_pnts = np.asarray(np.where(self.grid_map_grey == 255)).T
            #     for obj_id, obj in self.objects.items():
            #         object_grid = self.to_grid(obj.pos)
            #         if object_grid[1] >= self.height:
            #             continue
            #
            #         if object_grid[1] < 0:
            #             continue
            #
            #         if object_grid[0] < 0:
            #             continue
            #
            #         if object_grid[0] >= self.width:
            #             continue
            #
            #         if self.grid_map_grey[object_grid[1], object_grid[0]] == 0:
            #             distances = norm(free_pnts - np.flipud(object_grid),
            #                              axis=1)
            #             object_grid = np.flipud(free_pnts[np.argmin(
            #                 distances), :])
            #
            #         room_id = self.seg_map[object_grid[1], object_grid[0]]
            #         if room_id and room_id != 65280:
            #             if room_id not in self.room_ids_2_objects.keys():
            #                 self.room_ids_2_objects[room_id] = [obj_id]
            #             else:
            #                 self.room_ids_2_objects[room_id].append(obj_id)
            #
            #     for room_id in self.room_ids_2_objects:
            #         evidence = np.zeros(5, dtype='int')
            #         objects = self.room_ids_2_objects[room_id]
            #         for obj_id in objects:
            #             class_prob = self.objects[obj_id].class_probs
            #
            #             for i, evidence_class in enumerate(self.evidence_classes):
            #                 index = self.classes.index(evidence_class)
            #                 if class_prob[index] > 0.5:
            #                     evidence[i] = 1
            #
            #         req = self.c(evidence.tolist())
            #         self.room_ids_2_prob[room_id] = req.p
            #
            #     valid_pnts = np.asarray(np.where(np.logical_and(
            #         self.seg_map > 0, self.seg_map != 65280))).T
            #
            #     for i in range(1, self.frontier_num_labels):
            #         stat = self.frontier_stat[i]
            #         area = stat[cv2.CC_STAT_AREA]
            #         indices = (self.frontier_map == i)
            #         center = self.frontier_centroid[i].astype('int')
            #
            #         room_id = self.seg_map[center[1], center[0]]
            #
            #         if room_id == 0 or room_id == 65280:
            #             distances = norm(valid_pnts - np.flipud(center),
            #                              axis=1)
            #             center = np.flipud(valid_pnts[np.argmin(distances), :])
            #             room_id = self.seg_map[center[1], center[0]]
            #
            #         p = self.room_ids_2_prob.get(room_id)
            #         if not p:
            #             p = 0.5
            #         if area > 15:
            #             # semantic prior information
            #             self.frontier_map[indices] = area * p
            #             #self.frontier_map[indices] = 1
            #     end_time = time()

                image_message2 = self.bridge.cv2_to_imgmsg(
                    (self.frontier_map * 255 / np.amax(self.frontier_map)
                     ).astype(np.uint8), encoding="mono8")
                self.frontier_map_pub.publish(image_message2)

                #print('time spent waiting is', end_time - start_time)

            self.mutex.release()

    def find_sightings(self):
        self.sightings_map = {}
        self.sightings_phase = {}
        self.sightings_prob = {}

        max_line_len = 3 / self.resolution
        angles = np.arange(
            -np.pi, np.pi, step=1.0 / max_line_len, dtype=np.float32
        )

        self.obj_of_interest = None
        self.max_class_prob = -1.0
        target_class_index = self.classes.index(self.target_class)

        for obj_id, obj in self.objects.items():
            object_grid = self.to_grid(obj.pos)

            if object_grid[1] >= self.height:
                continue

            if object_grid[1] < 0:
                continue

            if object_grid[0] < 0:
                continue

            if object_grid[0] >= self.width:
                continue

            if self.grid_map_grey[object_grid[1], object_grid[0]] == 128:
                continue

            if obj.class_probs[target_class_index] > self.max_class_prob:
                self.max_class_prob = obj.class_probs[target_class_index]
                self.obj_of_interest = obj_id

        for obj_id in [self.obj_of_interest]:
            obj = self.objects[obj_id]
            object_grid = self.to_grid(obj.pos)
            start = np.flipud(object_grid).astype(int)
            object_sighting = np.zeros_like(self.grid_map_grey)
            object_phase = np.zeros_like(self.grid_map_grey)
            for angle in angles:
                end = start + max_line_len * np.array([np.cos(angle),
                                                       np.sin(angle)])
                end = np.rint(end).astype(np.int)
                discrete_line = list(zip(*line(start[0], start[1],
                                               end[0], end[1])))

                for pt in discrete_line:
                    x, y = pt

                    if norm(start - np.asarray(pt), ord=1) < 10:
                        continue

                    if x < 0 or x >= self.height:
                        break

                    if y < 0 or y >= self.width:
                        break

                    if self.grid_map_grey[x, y] != 255:
                        break

                    distance = norm(start - np.asarray(pt)) * self.resolution
                    if distance > 2:
                        object_sighting[x, y] = 1
                        object_phase[x, y] = -np.arctan2(start[0] - x,
                                                         start[1] - y)

            self.sightings_map[obj_id] = object_sighting
            self.sightings_phase[obj_id] = object_phase

        # obj_pose = obj.pos

        # center_coordinates = self.to_grid(obj_pose)
        sightings_map = (self.sightings_map[self.obj_of_interest]*255).astype('uint8')
        #cv2.circle(sightings_map, center_coordinates, 3, 255, -1)
        image_message = self.bridge.cv2_to_imgmsg(sightings_map,
                                                  encoding="mono8")
        self.sightings_map_pub.publish(image_message)

    def find_goal_region(self):
        self.goal_map = {}
        self.goal_phase = {}
        self.goal_prob = {}

        for obj_id in [self.obj_of_interest]:
            obj = self.objects[obj_id]
            object_grid = self.to_grid(obj.pos)
            object_goal = np.zeros_like(self.grid_map_grey)
            object_phase = np.zeros_like(self.grid_map_grey)
            cv2.circle(object_goal, object_grid, int(self.goal_radius/ self.resolution), 1, -1)
            object_goal[self.grid_map_grey != 255] = 0


            goals = np.nonzero(object_goal)
            object_phase[goals] = -np.arctan2(object_grid[1]- goals[0] , object_grid[0]- goals[1])

            self.goal_map[obj_id] = object_goal
            self.goal_phase[obj_id] = object_phase

        goal_map = (self.goal_map[self.obj_of_interest]*255).astype('uint8')
        #cv2.circle(sightings_map, center_coordinates, 3, 255, -1)
        image_message = self.bridge.cv2_to_imgmsg(goal_map, encoding="mono8")
        self.goal_map_pub.publish(image_message)


    def findFrontier(self):
        dx = [0, -1, -1, -1, 0, 1, 1, 1]
        dy = [1, 1, 0, -1, -1, -1, 0, 1]

        frontier_mat = cv2.Canny(self.grid_map_grey, 100, 200)

        grad_x = cv2.Scharr(self.grid_map_grey, cv2.CV_32F, 1, 0,
                            borderType=cv2.BORDER_ISOLATED)
        grad_y = cv2.Scharr(self.grid_map_grey, cv2.CV_32F, 0, 1,
                            borderType=cv2.BORDER_ISOLATED)

        free_pnts = np.asarray(np.where(frontier_mat == 255)).T.tolist()
        frontier_mat = np.zeros(np.shape(self.grid_map_grey), dtype=np.uint8)
        row, col = np.shape(self.grid_map_grey)

        for j in range(len(free_pnts)):
            r, c = free_pnts[j]
            if self.grid_map_grey[r, c] == 255:
                for i in range(8):
                    r1 = r + dx[i]
                    c1 = c + dy[i]

                    if 0 <= r1 < row and 0 <= c1 < col:
                        if self.grid_map_grey[r1, c1] == 128:
                            frontier_mat[r, c] = 255
                            break
            elif self.grid_map_grey[r, c] == 128:
                for i in range(8):
                    r1 = r + dx[i]
                    c1 = c + dy[i]

                    if 0 <= r1 < row and 0 <= c1 < col:
                        if self.grid_map_grey[r1, c1] == 255:
                            frontier_mat[r1, c1] = 255
                            # break

        # kernel = np.ones((5, 5), np.uint8) * 255
        # frontier_mat = cv2.dilate(frontier_mat, kernel, iterations=3)
        #
        # image_message = self.bridge.cv2_to_imgmsg(frontier_mat,
        #                                           encoding="mono8")
        # self.debug_pub.publish(image_message)

        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_mat, 8)

        self.frontier_num_labels = numLabels
        self.frontier_centroid = centroids
        self.frontier_stat = stats

        s = int(math.ceil(self.agent_radius / self.resolution))
        kernel = np.zeros((2 * s + 1, 2 * s + 1), np.uint8)
        cv2.circle(kernel, (s, s), s, 1, -1)

        obstacle_grids = (self.grid_map_grey == 0).astype('uint8')
        obstacle_grids = cv2.dilate(obstacle_grids, kernel, iterations=1,
                                    borderType=cv2.BORDER_ISOLATED).astype(
            bool)
        self.grid_map_grey[obstacle_grids] = 0

        for i in range(1, numLabels):
            stat = stats[i]
            area = stat[cv2.CC_STAT_AREA]
            indices = (labels == i)
            if area <= 15:
                frontier_mat[indices] = 0
                labels[indices] = 0

        frontier_mat[obstacle_grids] = 0
        labels[obstacle_grids] = 0

        grad_x[frontier_mat == 0] = 0
        grad_y[frontier_mat == 0] = 0

        iteration = 0
        iter_max = 2
        kernel = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).astype(np.uint8)
        frontier_new = np.zeros_like(frontier_mat)
        invalid_frontier = np.logical_or(obstacle_grids,
                                         self.grid_map_grey == 128)

        frontier_mat = labels.astype(float)
        while iteration < iter_max:
            frontier_new = cv2.dilate(frontier_mat, kernel, iterations=1,
                                      borderType=cv2.BORDER_ISOLATED)
            grad_x = cv2.filter2D(grad_x, ddepth=-1, kernel=kernel,
                                  borderType=cv2.BORDER_ISOLATED)
            grad_x = cv2.filter2D(grad_x, ddepth=-1, kernel=kernel,
                                  borderType=cv2.BORDER_ISOLATED)

            frontier_new[invalid_frontier] = 0
            grad_x[invalid_frontier] = 0
            grad_y[invalid_frontier] = 0

            iteration += 1
            if iteration == iter_max:
                frontier_new[np.logical_and(frontier_new > 0,
                                            frontier_mat > 0)] = 0
            frontier_mat = frontier_new

        # self.frontier_map = (frontier_mat == 255).astype(float)
        self.frontier_map = frontier_mat.astype(float)
        self.frontier_map[obstacle_grids] = 0
        for obj_id in self.sightings_map:
            self.sightings_map[obj_id][obstacle_grids] = 0
            # self.grid_map_color[self.sightings_map[obj_id] > 0, :] = [189, 114, 0]

        for obj_id in self.goal_map:
            self.goal_map[obj_id][obstacle_grids] = 0

        self.grid_map_color[self.frontier_map > 0, :] = [48, 172, 119]
        self.frontier_phase = -cv2.phase(grad_x, grad_y) + np.pi

    def semantic_map_callback(self, msg):
        if self.pos_var is None:
            rospy.loginfo('Robot pose covariance is not set.')
            return

        if self.pos is None:
            rospy.loginfo('Robot pose is not set.')
            return

        if self.grid_map_color is None:
            rospy.loginfo('Grid map is not constructed.')
            return

        grid_map = self.grid_map_color.copy()
        startAngle = 0
        endAngle = 360

        pos_grid = self.to_grid(self.pos)

        axesLength = (
        max(int(round(5 * np.sqrt(self.pos_var) / self.resolution)), 1),
        max(int(round(5 * np.sqrt(self.pos_var) / self.resolution)), 1))
        angle = 0
        # Red color in BGR
        color = (0, 0, 255)
        # Line thickness of 5 px
        thickness = 1
        cv2.ellipse(grid_map, pos_grid, axesLength,
                    angle, startAngle, endAngle, color, thickness)

        colors = [(142, 47, 126), (0, 255, 0)]
        # grid_map = np.ones_like(grid_map)*255
        for obj_msg in msg.objects:
            obj_id = obj_msg.id
            covariance = np.reshape(obj_msg.covariance, (2, 2))
            obj_pose = np.asarray([obj_msg.x, obj_msg.y])
            class_probs = np.asarray(obj_msg.probability)

            if obj_id not in self.objects:
                self.objects[obj_id] = MapObject(obj_id, obj_pose, covariance,
                                                 class_probs)
            else:
                obj = self.objects[obj_id]
                obj.update(obj_pose, covariance, class_probs)

            # if obj_id == self.obj_of_interest:
            #     center_coordinates = self.to_grid(obj_pose)
            #     color = colors[0]
            #
            #     w, v = eig(covariance)
            #     if w[0] < w[1]:
            #         axesLength = (
            #             max(int(round(5 * np.sqrt(w[1]) / self.resolution)), 1),
            #             max(int(round(5 * np.sqrt(w[0]) / self.resolution)), 1))
            #
            #         angle = np.arctan2(v[1, 1], v[0, 1]) * 180.0 / np.pi
            #     else:
            #         axesLength = (
            #             max(int(round(5 * np.sqrt(w[0]) / self.resolution)), 1),
            #             max(int(round(5 * np.sqrt(w[1]) / self.resolution)), 1))
            #
            #         angle = np.arctan2(v[1, 0], v[0, 0]) * 180.0 / np.pi
            #
            #     cv2.ellipse(grid_map, center_coordinates, axesLength,
            #                 angle, startAngle, endAngle, color,
            #                 thickness)

            # p2 = self.pos + 10 * self.resolution * np.asarray(
            #     [np.cos(self.ang),
            #      np.sin(self.ang)])
            # p2 = self.to_grid(p2)
            # .arrowedLine(grid_map, pos_grid, p2, color=(32, 177, 237))

        total_text_width = 0
        total_text_height = 0
        text_color_bg = (255, 255, 255)
        text_color = (0, 0, 0)

        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1  # 0.6
        font_thickness = 1

        for obj_id, obj in self.objects.items():
            if obj.visualized:
                text1 = 'object ID: ' + str(obj_id)

                text2 = 'pose: ' + str("%.2f" % obj.pos[0]) + ', ' + \
                        str("%.2f" % obj.pos[1])

                text3 = 'pose var: ' + str(
                    "%.2f" % (obj.pos_var[0, 0] * 10000.0)) + ', ' + \
                        str("%.2f" % (
                            obj.pos_var[0, 1] * 10000.0)) + ', ' + \
                        str("%.2f" % (
                            obj.pos_var[1, 0] * 10000.0)) + ', ' + \
                        str("%.2f" % (obj.pos_var[1, 1] * 10000.0))

                class_max = np.argmax(obj.class_probs)
                text4 = self.classes[class_max] + ': ' + str(
                    "%.2f" % obj.class_probs[class_max])

                text5 = ''

                texts = [text1, text2, text3, text4, text5]

                for text in texts:
                    text_size, _ = cv2.getTextSize(text, font, font_scale,
                                                   font_thickness)
                    text_w, text_h = text_size
                    total_text_height += (text_h + 1)
                    if text_w > total_text_width:
                        total_text_width = text_w

        if total_text_width and total_text_height:
            obj_info_image = np.zeros((total_text_height, total_text_width),
                                      dtype=np.uint8)
            text_coord = [0, 0]
            for obj_id, obj in self.objects.items():

                if obj.visualized:
                    text1 = 'object ID: ' + str(obj_id)

                    text2 = 'pose: ' + str("%.2f" % obj.pos[0]) + ', ' + \
                            str("%.2f" % obj.pos[1])

                    text3 = 'pose var: ' + str(
                        "%.2f" % (obj.pos_var[0, 0] * 10000.0)) + ', ' + \
                            str("%.2f" % (
                                obj.pos_var[0, 1] * 10000.0)) + ', ' + \
                            str("%.2f" % (
                                obj.pos_var[1, 0] * 10000.0)) + ', ' + \
                            str("%.2f" % (obj.pos_var[1, 1] * 10000.0))

                    class_max = np.argmax(obj.class_probs)
                    text4 = self.classes[class_max] + ': ' + str(
                        "%.2f" % obj.class_probs[class_max])

                    text5 = ''

                    texts = [text1, text2, text3, text4, text5]

                    for text in texts:
                        text_size, _ = cv2.getTextSize(text, font, font_scale,
                                                       font_thickness)
                        _, text_h = text_size
                        text_coord[1] = text_coord[1] + text_h + 1
                        cv2.putText(obj_info_image, text, tuple(text_coord),
                                    font,
                                    font_scale, 255, font_thickness)

            image_message = self.bridge.cv2_to_imgmsg(obj_info_image,
                                                      encoding="mono8")
            self.object_info_pub.publish(image_message)

        # if self.goal is not None:
        # cv2.drawMarker(grid_map, self.to_grid(self.goal),
        # color=(25, 83, 217), markerSize=10)
        # print('pos grid is ', pos_grid)
        # print('grid map shape is ', grid_map.shape)
        # if self.policy is not None:
        #     path = set()
        #     pos_grid_old = np.asarray(self.to_grid_old(self.pos))
        #     # print('pos is ', self.pos)
        #     # print('pos grid old is ', pos_grid_old)
        #     while True:
        #         grid_map[pos_grid[1], pos_grid[0], :] = [189, 114, 0]
        #         if tuple(pos_grid_old) in path:
        #             break
        #         else:
        #             path.add(tuple(pos_grid_old))
        #         pos_index = np.ravel_multi_index(np.flipud(pos_grid_old),
        #                                          (self.old_height,
        #                                           self.old_width))
        #         if self.policy.get(pos_index, -1) < 0:
        #             print('this grid has no policy')
        #             break
        #         else:
        #             action_index = self.policy[pos_index]
        #             pos_grid_old = pos_grid_old + np.flipud(
        #                 self.action_deltas[action_index])
        #             pos_map = self.to_map_old(pos_grid_old)
        #             pos_grid = self.to_grid(pos_map)

        # for obj_id in self.sightings_map:
        #     obj = self.objects[obj_id]
        #     obj_pose = obj.pos
        #
        #     center_coordinates = self.to_grid(obj_pose)
        #     color = colors[0]
        #
        #     w, v = eig(obj.pos_var)
        #     if w[0] < w[1]:
        #         axesLength = (
        #             max(int(round(5 * np.sqrt(w[1]) / self.resolution)), 1),
        #             max(int(round(5 * np.sqrt(w[0]) / self.resolution)), 1))
        #
        #         angle = np.arctan2(v[1, 1], v[0, 1])* 180.0 /np.pi
        #     else:
        #         axesLength = (
        #             max(int(round(5 * np.sqrt(w[0]) / self.resolution)), 1),
        #             max(int(round(5 * np.sqrt(w[1]) / self.resolution)), 1))
        #
        #         angle = np.arctan2(v[1, 0], v[0, 0])* 180.0 /np.pi
        #
        #     cv2.ellipse(grid_map, center_coordinates, axesLength,
        #                            angle, startAngle, endAngle, color,
        #                            thickness)

        image_message = self.bridge.cv2_to_imgmsg(grid_map,
                                                  encoding="bgr8")
        image_message.header.seq = self.seq
        image_message.header.stamp = rospy.get_rostime()
        image_message.header.frame_id = 'map'
        self.image_pub.publish(image_message)
        self.seq += 1


if __name__ == '__main__':
    rospy.init_node("planner")
    rospy.loginfo("Press Ctrl + C to terminate")
    whatever = Planner()
    try:
        whatever.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
