#!/usr/bin/env python

import gym
from gym import wrappers
import time
import datetime
import os
import json
import rospy

import qlearn

class QTrain:
    def start(self, testMode='test', caMode=False):
        task_and_robot_environment_name = rospy.get_param("/burger/task_and_robot_environment_name")
        env = gym.make(task_and_robot_environment_name)

        filepath = os.path.dirname(os.path.abspath(__file__))
        outdir = filepath + '/model/gazebo_gym_experiments/'
        path = filepath + '/model/burger_war_dqn_ep'

        dt_now = datetime.datetime.now()
        resultpath = filepath + '/model/' + dt_now.strftime('%Y-%m-%d-%H:%M:%S') + '.csv'

        resume_epoch = rospy.get_param("/burger/resume_epoch") # change to epoch to continue from
        resume_path = path + resume_epoch
        weights_path = resume_path + '_weights.json'
        params_json  = resume_path + '.json'

        if resume_epoch == "0":
            #Each time we take a sample and update our weights it is called a mini-batch.
            #Each time we run through the entire dataset, it's called an epoch.
            #PARAMETER LIST
            save_interval = rospy.get_param("/burger/save_interval")
            epochs = rospy.get_param("/burger/epochs")
            steps = rospy.get_param("/burger/steps")
            alpha = rospy.get_param("/burger/alpha")
            gamma = rospy.get_param("/burger/gamma")
            epsilon = rospy.get_param("/burger/epsilon")
            epsilon_discount = rospy.get_param("/burger/epsilon_discount")

            epsilon_decay = rospy.get_param("/burger/epsilon_decay")

            vel_max_x = rospy.get_param("/burger/vel_max_x")
            vel_min_x = rospy.get_param("/burger/vel_min_x")
            vel_max_z = rospy.get_param("/burger/vel_max_z")

            env = gym.wrappers.Monitor(env, outdir, force=True)
            qlearn = qlearn.QLearn(actions=range(env.action_space.n), alpha=alpha, gamma=gamma, epsilon=epsilon)
        # else:
        #     #Load weights, monitor info and parameter info.
        #     #ADD TRY CATCH fro this else
        #     with open(params_json) as outfile:
        #         d = json.load(outfile)
        #         save_interval = d.get('save_interval')
        #         epochs = d.get('epochs')
        #         steps = d.get('steps')
        #         if testMode == 'test':
        #             explorationRate = 0
        #         else:
        #             explorationRate = d.get('explorationRate')
        #         minibatch_size = d.get('minibatch_size')
        #         learnStart = d.get('learnStart')
        #         discountFactor = d.get('discountFactor')
        #         memorySize = d.get('memorySize')
        #         network_outputs = d.get('network_outputs')
        #         updateTargetNetwork = d.get('updateTargetNetwork')
        #         learningRate = d.get('learningRate')
        #         network_inputs = d.get('network_inputs')
        #         network_outputs = d.get('network_outputs')
        #         network_structure = d.get('network_structure')
        #         current_epoch = d.get('current_epoch')

        #         epsilon_decay = d.get('epsilon_decay')

        #         vel_max_x = d.get('vel_max_x')
        #         vel_min_x = d.get('vel_min_x')
        #         vel_max_z = d.get('vel_max_z')

        #     deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        #     deepQ.initNetworks(network_structure)

        #     deepQ.loadWeights(weights_path)

        #     if not os.path.exists(outdir):
        #         os.makedirs(outdir)
        #     self.clear_monitor_files(outdir)
        #     if not os.path.exists(monitor_path):
        #         os.makedirs(monitor_path)
        #     copy_tree(monitor_path,outdir)

        env._max_episode_steps = steps # env returns done after _max_episode_steps

        start_time = time.time()
        total_episodes = epochs

        for x in range(total_episodes):
            done = False

            cumulated_reward = 0 #Should going forward give more reward then L/R ?

            observation = env.reset()

            if qlearn.epsilon > 0.05:
                qlearn.epsilon *= epsilon_discount

            state = ''.join(map(str, observation))

            while not done:
                # Pick an action based on the current state
                action = qlearn.chooseAction(state)

                # Execute the action and get feedback
                observation, reward, done, info = env.step(action)
                cumulated_reward += reward

                nextState = ''.join(map(str, observation))

                qlearn.learn(state, action, reward, nextState)

                env._flush(force=True)

                if not(done):
                    state = nextState
                else:
                    break

            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

            if testMode == 'test':
                break

        env.close()
