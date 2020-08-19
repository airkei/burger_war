
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

import gazebo_env

import gym
from gym import utils, spaces
from gym.utils import seeding

import sys
sys.path.append(".")
from enemy_detector import ed


PI = math.pi

# Lidar
LIDAR_MAX_RANGE = 3.5
LIDAR_COLLISION_RANGE = 0.12

# MAP
MAP_WIDTH_X = 3.2
MAP_WIDTH_Y = 3.2

# Enemy Detection
ENEMY_MAX_DISTANCE = 0.7
ENEMY_MAX_DIRECTION = (2 * PI)
ENEMY_MAX_POINT = 20

# Point
POINT_NUM = 18
POINT_MYSELF_R_NAME = 'RE_R'
POINT_MYSELF_L_NAME = 'RE_L'
POINT_MYSELF_B_NAME = 'RE_B'
POINT_ENEMY_R_NAME = 'BL_R'
POINT_ENEMY_L_NAME = 'BL_L'
POINT_ENEMY_B_NAME = 'BL_B'

# Game
GAME_DURATION_SEC = 30

class BottiNodeEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)
        self._seed()

        self.var_reset()
        # Enemy Detector
        self.enemy_detector = ed.EnemyDetector()
        # Lidar callback
        rospy.Subscriber('scan', LaserScan, self.lidar_callback)
        # AMCL callback
        rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        # WarState callback
        rospy.Subscriber('war_state', String, self.war_state_callback)

    # Mode Setting
    def set_mode(self, runMode, collisionMode, outputs, vel_max_x, vel_min_x, vel_max_z):
        self.runMode = runMode
        self.collisionMode = collisionMode
        self.outputs = outputs

        self.vel_max_x = vel_max_x
        self.vel_min_x = vel_min_x
        self.vel_max_z = vel_max_z 

    # Reset
    def var_reset(self):
        self.sim_starttime = rospy.get_rostime()
        self.scan_time_prev = rospy.get_rostime()
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
        # WarState callback
        self.war_state = String()
        self.war_state_dict = {}
        self.war_state_dict_prev = {}

        self.war_state_myself_r_prev = 'n'
        self.war_state_myself_l_prev = 'n'
        self.war_state_myself_b_prev = 'n'
        self.war_state_enemy_r_prev = 'n'
        self.war_state_enemy_l_prev = 'n'
        self.war_state_enemy_b_prev = 'n'

        self.war_state_myself_r = 'n'
        self.war_state_myself_l = 'n'
        self.war_state_myself_b = 'n'
        self.war_state_enemy_r = 'n'
        self.war_state_enemy_l = 'n'
        self.war_state_enemy_b = 'n'

    def wait_for_topic(self, topic):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message(topic, LaserScan, timeout=5)
            except:
                pass
        return data

    def scan_env(self):
        env_list = []

        # Lidar(360)
        env_list.extend(self.scan)

        # Enemy Detection(4)
        # normalization(min:0/max:1)
        if self.is_near_enemy == True:
            is_near_enemy = 1
        else:
            is_near_enemy = 0

        enemy_direction = self.enemy_direction
        if self.enemy_direction is None:
            enemy_direction = 0
        enemy_direction /= ENEMY_MAX_DIRECTION

        enemy_dist = self.enemy_dist
        if self.enemy_dist is None:
            enemy_dist = 0
        enemy_dist /=  ENEMY_MAX_DISTANCE

        enemy_point = float(self.enemy_point)
        if enemy_point > ENEMY_MAX_POINT:
            enemy_point = ENEMY_MAX_POINT
        enemy_point /=  ENEMY_MAX_POINT
        env_list.extend([is_near_enemy, enemy_direction, enemy_dist, enemy_point])

        # War State(6)
        war_state = 0 * [6]
        if self.war_state_myself_r != 'n':  war_state[0] = 1
        if self.war_state_myself_l != 'n':  war_state[1] = 1
        if self.war_state_myself_b != 'n':  war_state[2] = 1
        if self.war_state_enemy_r  != 'n':  war_state[3] = 1
        if self.war_state_enemy_l  != 'n':  war_state[4] = 1
        if self.war_state_enemy_b  != 'n':  war_state[5] = 1
        env_list.extend(war_state)

        return env_list

    def calculate_observation(self, data):
        min_range = LIDAR_COLLISION_RANGE
        done = False
        points = []
        for i, distance in enumerate(data):
            if (min_range > distance > 0):
                done = True
                points.append(i)
        return data, done, points

### Geme System Function ###
    def is_game_timeout(self):
        if rospy.get_rostime().secs - self.sim_starttime.secs >= GAME_DURATION_SEC:
            print('[GAME]timeout')
            return True
        return False

    def is_game_called(self):
        try:
            if abs(self.war_state_dict['scores_b'] - self.war_state_dict['scores_r']) >= GAME_CALLED_SCORE:
                print('[GAME]called game')
                return True
        except:
            pass
        return False

    def is_get_enemy_points(self):
        if ((self.war_state_enemy_r != 'n') and (self.war_state_enemy_l != 'n') and (self.war_state_enemy_b != 'n')
            return True
        return False
### Geme System Function ###

### Callback ###
    def lidar_callback(self, data):
        self.scan = data.ranges
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

    def war_state_callback(self, data):
        self.war_state_dict = flatten(json.loads(data.data.replace('\n', '')))

        for i in range(POINT_NUM)
            if self.war_state_dict['targets_{}_name'.format(i)] == POINT_MYSELF_R_NAME:
                self.war_state_myself_r = self.war_state_dict['targets_{}_player'.format(i)]
            if self.war_state_dict['targets_{}_name'.format(i)] == POINT_MYSELF_L_NAME:
                self.war_state_myself_l = self.war_state_dict['targets_{}_player'.format(i)]
            if self.war_state_dict['targets_{}_name'.format(i)] == POINT_MYSELF_B_NAME:
                self.war_state_myself_b = self.war_state_dict['targets_{}_player'.format(i)]

            if self.war_state_dict['targets_{}_name'.format(i)] == POINT_ENEMY_R_NAME:
                self.war_state_enemy_r = self.war_state_dict['targets_{}_player'.format(i)]
            if self.war_state_dict['targets_{}_name'.format(i)] == POINT_ENEMY_L_NAME:
                self.war_state_enemy_l = self.war_state_dict['targets_{}_player'.format(i)]
            if self.war_state_dict['targets_{}_name'.format(i)] == POINT_ENEMY_B_NAME:
                self.war_state_enemy_b = self.war_state_dict['targets_{}_player'.format(i)]
### Callback ###

### Gym Functions ###
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # move
        vel_minus = False
        if action >= self.outputs/2:
            vel_minus = True
            action -= self.outputs/2

        ang_vel = ((action//2 - self.outputs//8) * self.vel_max_z) / (self.outputs//8)

        vel_cmd = Twist()
        if (action % 2) == 0:
            vel_cmd.linear.x = self.vel_max_x
        else:
            vel_cmd.linear.x = self.vel_min_x
        if vel_minus:
            vel_cmd.linear.x *= -1

        vel_cmd.angular.z = ang_vel
        print(action, vel_cmd.linear.x, vel_cmd.angular.z)
        self.vel_pub.publish(vel_cmd)

        # wait 400ms
        if (rospy.get_rostime() - self.scan_time_prev) <= rospy.Duration(0.3):
            # dummy read
            self.wait_for_topic('/scan')
        data = self.wait_for_topic('/scan')
        self.scan = data.ranges
        self.scan_time_prev = rospy.get_rostime()

        # scan
        state = self.scan_env()
        _, collision, points = self.calculate_observation(self.scan)

        # vel/collistion reward
        reward = 0
        if (collision or (self.enemy_dist is None)):
            self.collision_cnt = self.collision_cnt + 1
            reward -= 5
        else:
            self.collision_cnt = 0
            reward += (9.6 / self.enemy_dist)

        # point reward
        if ((self.war_state_myself_r != 'n') and (self.war_state_myself_r != self.war_state_myself_r_prev)):
            reward -= 30
        if ((self.war_state_myself_l != 'n') and (self.war_state_myself_l != self.war_state_myself_l_prev)):
            reward -= 30
        if ((self.war_state_myself_b != 'n') and (self.war_state_myself_b != self.war_state_myself_b_prev)):
            reward -= 50
        if ((self.war_state_enemy_r != 'n') and (self.war_state_enemy_r != self.war_state_enemy_r_prev)):
            reward += 30
        if ((self.war_state_enemy_l != 'n') and (self.war_state_enemy_l != self.war_state_enemy_l_prev)):
            reward += 30
        if ((self.war_state_enemy_b != 'n') and (self.war_state_enemy_b != self.war_state_enemy_b_prev)):
            reward += 50
        self.war_state_myself_r_prev = self.war_state_myself_r
        self.war_state_myself_l_prev = self.war_state_myself_l
        self.war_state_myself_b_prev = self.war_state_myself_b
        self.war_state_enemy_r_prev = self.war_state_enemy_r
        self.war_state_enemy_l_prev = self.war_state_enemy_l
        self.war_state_enemy_b_prev = self.war_state_enemy_b

        # check game end
        done = self.is_game_timeout() or self.is_game_called()

        critical = len(points) > 0 and ((min(points) <= 45) or (max(points) >= 315))
        if ((self.collision_cnt >= 6) or critical):
            self.collision_cnt = 0
            reward -= 100
            done = True

            # emergency recovery
            if self.runMode == 'test':            
                for _ in range(12):
                    vel_cmd.linear.x = -self.vel_max_x
                    vel_cmd.angular.z = 0
                    self.vel_pub.publish(vel_cmd)
                    data = self.wait_for_topic('/scan')
                    self.scan = data.ranges
                state = self.scan_env()

        rospy.loginfo('action:' + str(action) + ', reward:' + str(reward))

        if self.runMode == 'test':
            done = False

        return np.asarray(state), reward, done, {}

    def reset(self):
        if self.runMode == 'test':
            self.var_reset()

            data = self.wait_for_topic('/scan')
            self.scan = data.ranges
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

                # reset variables
                self.var_reset()

                data = self.wait_for_topic('/scan')
                self.scan = data.ranges

        state = self.scan_env()

        return np.asarray(state)
### Gym Functions ###
