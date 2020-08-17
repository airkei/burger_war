
import json
import math

from flatten_json import flatten
import numpy as np
import requests

import rospy
import tf
from std_srvs.srv import Empty
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from opencv_apps.msg import CircleArrayStamped

import gazebo_env

import gym
from gym import utils, spaces
from gym.utils import seeding

import sys
sys.path.append(".")
from enemy_detector import ed


# Lidar
LIDAR_MAX_RANGE = 3.5
LIDAR_COLLISION_RANGE = 0.12

# For Enemy Camera Detection
ENEMY_CAM_THRESHOLD_Y = 280

# ACTION
ACTION_FORWARD = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2


class BottiNodeEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self._seed()

        self.var_reset()
        # Enemy Detector
        self.enemy_detector = ed.EnemyDetector()
        # Lidar callback
        rospy.Subscriber('scan', LaserScan, self.lidar_callback)
        # AMCL callback
        rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        # Camera
        rospy.Subscriber('hough_circles/circles', CircleArrayStamped, self.camera_callback)
        # WarState callback
        rospy.Subscriber('war_state', String, self.war_state_callback)

    # Mode Setting
    def set_mode(self, testMode, caMode, vel_max_x, vel_min_x, vel_max_z, scan_points):
        self.testMode = testMode
        self.caMode = caMode

        self.vel_max_x = vel_max_x
        self.vel_min_x = vel_min_x
        self.vel_max_z = vel_max_z
        self.scan_points = scan_points

    # Reset
    def var_reset(self):
        self.sim_starttime = rospy.get_rostime()
        self.scan_time_prev = rospy.get_rostime()
        self.prev_score = 0
        self.collision_cnt = 0

        # Enemy Detector
        self.is_near_enemy = 0
        self.enemy_direction = 0
        self.enemy_dist = 0
        self.enemy_point = 0
        # Lidar callback
        self.scan = []
        # AMCL callback
        self.pose_x = 0
        self.pose_y = 0
        self.th = 0
        # Camera
        self.enemy_cam_detect = 0
        self.enemy_cam_pose_x = 0
        self.enemy_cam_pose_y = 0
        # WarState callback
        self.war_state = String()
        self.war_state_dict = {}
        self.war_state_dict_prev = {}

    def wait_for_topic(self, topic):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(topic, LaserScan, timeout=5)
            except:
                pass
        return data

    def calculate_observation(self, data):
        min_range = LIDAR_COLLISION_RANGE
        done = False
        for distance in data:
            if (min_range > distance > 0):
                done = True
        return data, done

    def rounding(self, val):
        # x10
        data = int(val * 10)
        # rounding( % 5)
        data -= data % 5
        # rounding(* 2 / 10)
        data = (data * 2) / 10

        return data

    def discretize_observation(self, data):
        discretized_ranges = []
        min_range = LIDAR_COLLISION_RANGE
        done = False

        points = []
        if (self.scan_points % 2) != 0:
            points.append(0)
            
        for i in range(self.scan_points//2):
            points.append(90/(self.scan_points//2) * (i+1))

        for i in range(self.scan_points//2):
            points.append(360 -90/(i+1))

        for i, item in enumerate(data.ranges):
            if i in points:
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    discretized_ranges.append(self.rounding(LIDAR_MAX_RANGE))
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(self.rounding(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges, done

### Geme System Function ###
    def is_game_timeout(self):
        if rospy.get_rostime().secs - self.sim_starttime.secs >= 180:
            print('[GAME]timeout')
            return True
        return False

    def is_game_called(self):
        if abs(self.war_state_dict['scores_b'] - self.war_state_dict['scores_r']) >= 10:
            print('[GAME]called game')
            return True
        return False
### Geme System Function ###

### Callback ###
    def lidar_callback(self, data):
        # self.scan = data.ranges
        # enemy detection
        self.is_near_enemy, self.enemy_direction, self.enemy_dist, self.enemy_point = self.enemy_detector.findEnemy(data.ranges, self.pose_x, self.pose_y, self.th)

# respect level_3_clubhouse freom burger_war
# https://github.com/OneNightROBOCON/burger_war
    def pose_callback(self, data):
        self.pose_x = data.pose.pose.position.x
        self.pose_y = data.pose.pose.position.y
        quaternion = data.pose.pose.orientation
        rpy = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
        self.th = rpy[2]
# End Respect

    def camera_callback(self, data):
        self.enemy_cam_detect = 0
        self.enemy_cam_pose_x = 0
        self.enemy_cam_pose_y = 0

        for circle in data.circles:
            if circle.center.y >= ENEMY_CAM_THRESHOLD_Y:
                self.enemy_cam_detect = 1
                self.enemy_cam_pose_x = circle.center.x
                self.enemy_cam_pose_y = circle.center.y
                break

    def war_state_callback(self, data):
        self.war_state_dict = flatten(json.loads(data.data.replace('\n', '')))

### Callback ###

### Gym Functions ###
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # move
        vel_cmd = Twist()
        if action == ACTION_FORWARD: #FORWARD
            vel_cmd.linear.x = self.vel_max_x
            vel_cmd.angular.z = 0.0
        elif action == ACTION_LEFT: #LEFT
            vel_cmd.linear.x = self.vel_min_x
            vel_cmd.angular.z = self.vel_max_z
        elif action == ACTION_RIGHT: #RIGHT
            vel_cmd.linear.x = self.vel_min_x
            vel_cmd.angular.z = -self.vel_max_z

        self.vel_pub.publish(vel_cmd)

        # scan
        data = self.wait_for_topic('/scan')
        self.scan, collision = self.discretize_observation(data)

        # vel/collistion reward
        reward = 0
        if collision:
            self.collision_cnt = self.collision_cnt + 1
            reward -= 5
        else:
            self.collision_cnt = 0
            if action == ACTION_FORWARD: 
                reward += 5
            else:
                reward += 1

        # check game end
        done = self.is_game_timeout() or self.is_game_called()
        if self.collision_cnt >= 10:
            self.collision_cnt = 0
            reward -= 200
            done = True

            # emergency recovery
            for _ in range(12):
                vel_cmd.linear.x = -self.vel_max_x
                vel_cmd.angular.z = 0
                self.vel_pub.publish(vel_cmd)
                data = self.wait_for_topic('/scan')
            self.scan, _ = self.discretize_observation(data)

        rospy.loginfo('action:' + str(action) + ', reward:' + str(reward))

        if self.testMode == 'test':
            done = False

        return self.scan, reward, done, {}

    def reset(self):
        if self.testMode == 'test':
            self.var_reset()

            data = self.wait_for_topic('/scan')
            self.scan, _ = self.discretize_observation(data)
            self.wait_for_topic('/war_state')
        else:
            for _ in range(2):        
                # Resets the state of the environment and returns an initial observation.
                rospy.wait_for_service('/gazebo/reset_simulation')
                try:
                    #reset_proxy.call()
                    self.reset_proxy()
                except (rospy.ServiceException) as e:
                    print ("/gazebo/reset_simulation service call failed")
        
                # reset judge server
                requests.get('http://localhost:5000/teambt_initialize')

                # reset AMCL pose
                pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=10)
                pose = PoseWithCovarianceStamped()
                pose.header.frame_id = "/map"
                pose.pose.pose.position.x=-1.3
                pose.pose.pose.orientation.w=1.0
                pub.publish(pose)

                # resert variables
                self.var_reset()

                data = self.wait_for_topic('/scan')
                self.scan, _ = self.discretize_observation(data)
                self.wait_for_topic('/war_state')

        return self.scan
### Gym Functions ###
