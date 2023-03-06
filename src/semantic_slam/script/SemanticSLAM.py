#!/usr/bin/env python
import numpy as np
from numpy.linalg import inv, norm, eig
import rospy
import tf2_ros
from sensor_msgs.msg import Image, CameraInfo
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes, RangeBearing, \
    RangeBearings,  Object2D, Object2DArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from tf2_ros import TransformException, ConnectivityException, ExtrapolationException
import tf
from scipy.stats import dirichlet
from scipy.stats import entropy
import pickle

class MapObject:
    def __init__(self, object_id, pos, pos_var, class_probs, obj_class):
        self.id = object_id
        self.pos = pos
        self.pos_var = pos_var
        self.class_probs = class_probs
        self.obj_class = obj_class

    def update(self, pos, pos_var, class_probs):
        self.pos = pos
        self.pos_var = pos_var
        self.class_probs = class_probs

class SemanticSLAM():
    def __init__(self):
        self.target_frame = "camera_link"
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.ang_var = None
        self.pos_var = None

        self.range_var = 8e-04
        self.bearing_var = 0.01
        self.sigma_delta = np.diag([self.range_var, self.bearing_var])
        self.sigma_delta_inv = inv(self.sigma_delta)

        self.ang = None
        self.pos = None
        self.sigma_p = None
        self.sigma_p_inv = None

        self.alpha_constant = 1.1

        self.classes = ['towel', 'objects', 'lighting', 'stool', 'counter',
                           'door', 'clothes', 'appliances', 'furniture',
                           'shelving', 'bed', 'blinds', 'table', 'cabinet',
                           'shower', 'chair', 'chest_of_drawers', 'tv_monitor',
                           'toilet', 'mirror', 'sofa', 'cushion', 'sink',
                           'banana', 'cheezit']
        self.nC = len(self.classes)

        self.odom_sub = rospy.Subscriber("/rtabmap/odom", Odometry, self.odom_callback)
        self.range_sub = rospy.Subscriber("/range_bearing", RangeBearings, self.range_callback)
        self.map_pub = rospy.Publisher("/semantic_map", Object2DArray, queue_size=10)
        self.t = 0
        self.t_series = []
        self.entropy_series = []
        self.A_opt = []
        self.D_opt = []
        self.E_opt = []

        self.objects = {}
        self.time_id = 0

        rospy.on_shutdown(self.save_data)

    def odom_callback(self, msg):
        # only takes the covariance, the pose is taken from tf transformation
        self.pos_var = msg.pose.covariance[0]
        self.ang_var = msg.pose.covariance[-1]
        self.sigma_p = np.diag([self.pos_var, self.pos_var, self.ang_var])
        self.sigma_p_inv = inv(self.sigma_p)
        self.t = msg.header.stamp.to_sec()

    def range_callback(self, msg):
        if self.pos_var is None:
            rospy.loginfo('robot pose covariance is not set')
            return

        from_frame_rel = self.target_frame
        to_frame_rel = 'map'

        try:
            trans = self.tfBuffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rospy.Time(0))

            self.pos = np.asarray([trans.transform.translation.x,
                                    trans.transform.translation.y])

            self.ang = tf.transformations.euler_from_quaternion([trans.transform.rotation.x,
                                                                 trans.transform.rotation.y,
                                                                 trans.transform.rotation.z,
                                                                 trans.transform.rotation.w])[-1]

        except (TransformException,ConnectivityException, ExtrapolationException) as ex:
            rospy.loginfo('Could not transform %s to %s: ', to_frame_rel,
                          from_frame_rel)
            return

        for range_bearing in msg.range_bearings:
            obj_range = range_bearing.range
            bearing = range_bearing.bearing
            obj_id = range_bearing.id
            probability = range_bearing.probability
            obj_class = range_bearing.obj_class

            if obj_id not in self.objects:

                ux = self.pos[0]
                uy = self.pos[1]

                mx = ux + np.cos(self.ang + bearing)*obj_range
                my = uy + np.sin(self.ang + bearing)*obj_range

                x_var = self.pos_var + \
                    np.power(np.sin(self.ang + bearing)*obj_range, 2) *\
                    (self.ang_var + self.bearing_var) + \
                    np.power(np.cos(self.ang + bearing),       2) *\
                    self.range_var

                y_var = self.pos_var + \
                    np.power(np.cos(self.ang + bearing)*obj_range, 2) *\
                    (self.ang_var + self.bearing_var) + \
                    np.power(np.sin(self.ang + bearing),       2) *\
                    self.range_var

                object_pos = np.asarray([mx, my])
                object_pos_var = np.diag([x_var, y_var])

                class_probs = []
                for i in range(self.nC):
                    alpha = np.ones(self.nC)
                    alpha[i] = self.alpha_constant
                    class_probs.append(dirichlet.pdf(probability, alpha))

                class_probs = np.asarray(class_probs)
                class_probs = np.asarray(class_probs) / np.sum(class_probs)

                self.objects[obj_id] = MapObject(obj_id, object_pos,
                                                 object_pos_var, class_probs, obj_class)

            else:
                obj = self.objects[obj_id]

                obj_pos = obj.pos
                class_probs = obj.class_probs
                obj_x = obj_pos[0]
                obj_y = obj_pos[1]

                sigma_m = obj.pos_var
                sigma_m_inv = inv(sigma_m)

                x = self.pos[0]
                y = self.pos[1]

                d = norm(np.asarray([x, y]) - obj_pos)
                K1 = np.asarray([[(obj_x - x)/d,       (obj_y - y)/d],
                                 [(y - obj_y)/(d**2), (obj_x - x)/(d**2)]])

                K2 = np.asarray([[(x - obj_x)/d,      (y - obj_y)/d,      0],
                                 [(obj_y - y)/(d**2), (x - obj_x)/(d**2), -1]])
                z = np.asarray([obj_range, bearing])

                # print('robot pose is ', [x, y, self.ang])
                # print('old object pose is ', obj.pos)
                # print(np.arctan2(obj_y-y, obj_x-x))
                dz = z - np.asarray([d, np.arctan2(obj_y-y, obj_x-x) - self.ang])
                dz[1] = np.arctan2(np.sin(dz[1]), np.cos(dz[1]))

                # print('robot pose covariance is ', self.sigma_p)
                # print('old object pose covariance is ', obj.pos_var)
                #
                # print('z is ', z)
                # print('dz is ', dz)

                psi_inv = np.matmul(np.matmul(K2.T, self.sigma_delta_inv), K2) \
                    + self.sigma_p_inv
                psi = inv(psi_inv)

                M1 = self.sigma_delta_inv - np.matmul(np.matmul(np.matmul(np.matmul(self.sigma_delta_inv, K2),
                                                                          psi),
                                                                K2.T),
                                                      self.sigma_delta_inv)

                updated_pos_var = inv(np.matmul(np.matmul(K1.T, M1), K1) +
                                      sigma_m_inv)

                K = np.matmul(np.matmul(updated_pos_var, K1.T), M1)
                updated_pos = obj_pos + np.matmul(K, dz)

                # print('updated pose is ', updated_pos)
                # print('updated pose covariance is', updated_pos_var)

                for i in range(self.nC):
                    alpha = np.ones(self.nC)
                    alpha[i] = self.alpha_constant
                    class_probs[i] = dirichlet.pdf(probability, alpha) * \
                                     class_probs[i]

                class_probs = np.asarray(class_probs)
                class_probs = np.asarray(class_probs) / np.sum(class_probs)
                #class_probs = (class_probs + 0.004)/(class_probs + 0.004*len(self.classes))

                obj.update(updated_pos, updated_pos_var, class_probs)

        semantic_map_msg = Object2DArray()
        semantic_map_msg.header = msg.header
        objects = []
        for obj_id in self.objects:
            obj_msg = Object2D()
            obj = self.objects[obj_id]

            obj_msg.x = obj.pos[0]
            obj_msg.y = obj.pos[1]

            obj_msg.covariance = obj.pos_var.flatten()
            obj_msg.id = obj_id
            obj_msg.probability = obj.class_probs.tolist()
            objects.append(obj_msg)

        # object_num = len(self.objects)
        # average_entropy = 0
        # A_opt = 0
        # D_opt = 0
        # E_opt = 0
        #
        # file1 = open("/home/bear/Github/habitat-lab/offline_map/" + str(self.time_id) +  ".txt", "w")
        file1 = open("/home/bear/Github/habitat-lab/" + "current_map.txt", "w")
        file1.write(str(self.t) + '\n')
        for obj_id in self.objects:
            obj = self.objects[obj_id]
            # average_entropy += entropy(obj.class_probs, base=2)
            # w, v = eig(obj.pos_var)

            # A_opt += w.sum()
            # D_opt += w.prod()
            # E_opt += w.max()

            line = str(obj_id) + ' ' + str(obj.pos[0]) + ' ' + str(obj.pos[1])
            for conv in obj.pos_var.flatten():
                line = line + ' ' + str(conv)

            for prob in obj.class_probs:
                line = line + ' ' + str(prob)

            file1.write(line+' ' + obj.obj_class + '\n')
        file1.close()
        #
        # A_opt /= object_num
        # D_opt /= object_num
        # E_opt /= object_num
        #
        # self.A_opt.append(A_opt)
        # self.D_opt.append(D_opt)
        # self.E_opt.append(E_opt)
        #
        # average_entropy = average_entropy / object_num
        #
        # self.t_series.append(self.t)
        #
        # self.entropy_series.append(average_entropy)
        semantic_map_msg.objects = objects
        self.map_pub.publish(semantic_map_msg)


        self.time_id += 1

    def save_data(self):
        # with open("/home/bear/Github/habitat-lab/log.pkl", "wb") as fp:  # Pickling
        #     pickle.dump([self.t_series, self.entropy_series], fp)

        # with open('/home/bear/Github/habitat-lab/log.npy', 'wb') as f:
        #     np.save(f, self.t_series)
        #     np.save(f, self.entropy_series)
        #     np.save(f, self.A_opt)
        #     np.save(f, self.D_opt)
        #     np.save(f, self.E_opt)
        pass

if __name__ == '__main__':
    rospy.init_node("semantic_SLAM")
    rospy.loginfo("Press Ctrl + C to terminate")
    whatever = SemanticSLAM()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
