#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

import dqn_modules
from dqn_modules import train


if __name__ == '__main__':
    rospy.init_node('dqn_run')
    dqntrain = train.Train()

    # Only for collision avoidance
    # dqntrain.start(testMode='pretrain', caMode=True) # For pre-training(manual)
    # dqntrain.start(testMode='train', caMode=True) # For training
    dqntrain.start(testMode='test', caMode=True) # For test

    # For Production
    # dqntrain.start(testMode='pretrain', caMode=False) # For pre-training(manual)
    # dqntrain.start(testMode='train', caMode=False) # For training
    # dqntrain.start(testMode='test', caMode=False) # For test
