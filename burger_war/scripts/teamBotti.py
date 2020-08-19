#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

import dqn_modules
from dqn_modules import train

# import qlearn_modules
# from qlearn_modules import train


if __name__ == '__main__':
    side = sys.argv[1]
    print('side = ' + side)

    rospy.init_node('botti')
    training = train.Train()

    # Only for collision avoidance
    # training.start(runMode='train', collisionMode=True) # For training
    # training.start(runMode='test', collisionMode=True) # For test

    # For Production
    training.start(runMode='train', collisionMode=False, side=side) # For training
    # training.start(runMode='test', collisionMode=False) # For test

    # For Battle
    # training.start(runMode='train', battleMode=True) # For training
    # training.start(runMode='test', battleMode=False) # For test