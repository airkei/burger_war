#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

# import dqn_modules
# from dqn_modules import train

import qlearn_modules
from qlearn_modules import train


if __name__ == '__main__':
    rospy.init_node('botti')
    dqntrain = train.Train()

    # Only for collision avoidance
    dqntrain.start(testMode='train', caMode=True) # For training
    # dqntrain.start(testMode='test', caMode=True) # For test

    # For Production
    # dqntrain.start(testMode='train', caMode=False) # For training
    # dqntrain.start(testMode='test', caMode=False) # For test
