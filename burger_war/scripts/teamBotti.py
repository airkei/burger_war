#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import rospy

import dqn_modules
from dqn_modules import train

if __name__ == '__main__':
    side = sys.argv[1]
    print('side = ' + side)

    rospy.init_node('botti')

    # Only for collision avoidance
    # training = train.Train(side=side, runMode='train', collisionMode=True) # For training
    # training = train.Train(side=side, runMode='test',  collisionMode=True) # For test

    # For Production
    # training = train.Train(side=side, runMode='train', collisionMode=False) # For training
    training = train.Train(side=side, runMode='test',  collisionMode=False) # For test

    training.start()
