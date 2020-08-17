#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

# import dqn_modules
# from dqn_modules import train

import qlearn_modules
from qlearn_modules import train


if __name__ == '__main__':
    rospy.init_node('botti')
    training = train.Train()

    # Only for collision avoidance
    training.start(testMode='train', caMode=True) # For training
    # training.start(testMode='test', caMode=True) # For test

    # For Production
    # training.start(testMode='train', caMode=False) # For training
    # training.start(testMode='test', caMode=False) # For test
