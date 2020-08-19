
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

# Game
GAME_DURATION_SEC = 180
GAME_CALLED_SCORE = 10

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
    def set_mode(self, side, runMode, collisionMode, outputs, vel_max_x, vel_min_x, vel_max_z):
        self.side = side
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
        self.prev_score_r = 0
        self.prev_score_b = 0
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

        # Lidar(181) -90 to 90 degree
        # scan = self.scan
        first = np.array(self.scan[1:91])
        last = np.array(self.scan[270:360])

        scan = []
        scan.extend(last)
        scan.append(self.scan[0])
        scan.extend(first)

        # normalization(min:0/max:1)
        npscan = np.array(scan)
        npscan[np.isinf(npscan)] = LIDAR_MAX_RANGE
        npscan = npscan / LIDAR_MAX_RANGE
        env_list = npscan.tolist()

        # AMCL(3)
        # normalization(min:0/max:1)
        pose_x = (self.pose_x + MAP_WIDTH_X/2) / MAP_WIDTH_X
        pose_y = (self.pose_y + MAP_WIDTH_Y/2) / MAP_WIDTH_Y
        th = (self.pose_y + PI) / (2 * PI) 
        env_list.extend([pose_x, pose_y, th])

        if not self.collisionMode:
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

            # War State(18)
            war_state = [0] * 18
            for i in range(0, 18):
                # one-hot encoding
                try:
                    if self.war_state_dict['targets_{}_player'.format(i)] == 'r':
                        war_state[i] = 1
                    else:
                        war_state[i] = 0
                except:
                    war_state[i] = 0
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
### Callback ###

### Gym Functions ###
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # move
        ang_vel = ((action//2 - self.outputs//4) * self.vel_max_z) / (self.outputs//4)

        vel_cmd = Twist()
        if (action % 2) == 0:
            vel_cmd.linear.x = self.vel_max_x
        else:
            vel_cmd.linear.x = self.vel_min_x 
        vel_cmd.angular.z = ang_vel
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

        # reward
        reward = 0
        if collision:
            self.collision_cnt = self.collision_cnt + 1
        else:
            self.collision_cnt = 0
            # velocity reward
            reward += 30 * abs(vel_cmd.linear.x)

            # map reward
            if (-0.5 <= self.pose_x <= 0.5) and (-0.5 <= self.pose_y <= 0.5):
                reward += 2

        # point reward
        if not self.collisionMode: # production mode
            try:
                if self.side == 'r':
                    reward += (self.war_state_dict['scores_r'] - self.prev_score_r) * 10
                    reward -= (self.war_state_dict['scores_b'] - self.prev_score_b) * 10
                else:
                    reward -= (self.war_state_dict['scores_r'] - self.prev_score_r) * 10
                    reward += (self.war_state_dict['scores_b'] - self.prev_score_b) * 10

                self.prev_score_r = self.war_state_dict['scores_r']
                self.prev_score_b = self.war_state_dict['scores_b']
            except:
                pass

        # check game end
        done = self.is_game_timeout() or self.is_game_called()

        critical = len(points) > 0 and ((min(points) <= 45) or (max(points) >= 315))
        if ((self.collision_cnt >= 6) or critical):
            self.collision_cnt = 0
            reward -= 200
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
