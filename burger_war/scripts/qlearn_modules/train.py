#!/usr/bin/env python

import gym
from gym import wrappers
import time
import datetime
import os
import json
import rospy

import qlearn

class Train:
    def start(self, testMode='test', caMode=False):
        task_and_robot_environment_name = rospy.get_param("/burger/task_and_robot_environment_name")
        env = gym.make(task_and_robot_environment_name)

        filepath = os.path.dirname(os.path.abspath(__file__))
        outdir = filepath + '/model/gazebo_gym_experiments/'
        path = filepath + '/model/burger_war_qlearn_ep'

        dt_now = datetime.datetime.now()
        resultpath = filepath + '/model/' + dt_now.strftime('%Y-%m-%d-%H:%M:%S') + '.csv'

        resume_epoch = rospy.get_param("/burger/resume_epoch") # change to epoch to continue from
        resume_path = path + resume_epoch
        weights_path = resume_path + '.csv'
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
            scan_points = rospy.get_param("/burger/scan_points")
            actions = rospy.get_param("/burger/actions")

            vel_max_x = rospy.get_param("/burger/vel_max_x")
            vel_min_x = rospy.get_param("/burger/vel_min_x")
            vel_max_z = rospy.get_param("/burger/vel_max_z")

            env = gym.wrappers.Monitor(env, outdir, force=True)
            ql = qlearn.QLearn(actions=actions, alpha=alpha, gamma=gamma, epsilon=epsilon)
        else:
            #Load weights, monitor info and parameter info.
            #ADD TRY CATCH fro this else
            with open(params_json) as outfile:
                d = json.load(outfile)
                save_interval = d.get('save_interval')
                epochs = d.get('epochs')
                steps = d.get('steps')
                if testMode == 'test':
                    epsilon = 0
                else:
                    epsilon = d.get('epsilon')
                alpha = d.get('alpha')
                gamma = d.get('gamma')
                if testMode == 'test':
                    epsilon = 0
                else:
                    epsilon = d.get('epsilon')
                epsilon_discount = d.get('epsilon_discount')
                scan_points = d.get('scan_points')
                actions = d.get('actions')

                vel_max_x = d.get('vel_max_x')
                vel_min_x = d.get('vel_min_x')
                vel_max_z = d.get('vel_max_z')

            ql = qlearn.QLearn(actions=actions, alpha=alpha, gamma=gamma, epsilon=epsilon)
            ql.loadWeights(weights_path)

        env._max_episode_steps = steps # env returns done after _max_episode_steps

        start_time = time.time()
        total_episodes = epochs

        env.set_mode(testMode, caMode, vel_max_x, vel_min_x, vel_max_z, scan_points)
        for x in range(total_episodes):
            done = False

            cumulated_reward = 0 #Should going forward give more reward then L/R ?

            observation = env.reset()

            if ql.epsilon > 0.05:
                ql.epsilon *= epsilon_discount

            state = ''.join(map(str, observation))

            episode_step = 0
            while not done:
                # Pick an action based on the current state
                action = ql.chooseAction(state)

                # Execute the action and get feedback
                observation, reward, done, info = env.step(action)
                cumulated_reward += reward

                nextState = ''.join(map(str, observation))

                ql.learn(state, action, reward, nextState)

                env._flush(force=True)

                episode_step += 1

                if not done:
                    state = nextState
                else:
                    break

            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            print ("EP: " + str(x+1) + " - [alpha: "+str(round(ql.alpha, 2)) + " - gamma: " + str(round(ql.gamma, 2)) + " - epsilon: " + str(round(ql.epsilon, 2)) + "] - Reward: " + str(cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s))

            if (episode_step % save_interval) == 0:
                #save model weights and monitoring data every save_interval epochs.
                ql.saveModel(path + str(x) + '.csv')
                #save simulation parameters.
                parameter_keys = ['save_interval', 'epochs','steps','alpha','gamma','epsilon','epsilon_discount','scan_points','actions','vel_max_x','vel_min_x','vel_max_z']
                parameter_values = [save_interval, epochs, steps, ql.alpha, ql.gamma, ql.epsilon, epsilon_discount, scan_points, actions, vel_max_x, vel_min_x, vel_max_z]
                parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                with open(path + str(x) + '.json', 'w') as outfile:
                    json.dump(parameter_dictionary, outfile)

            with open(resultpath, mode='a') as f:
                f.write(str(x) + "," + format(episode_step + 1) + "," + str(cumulated_reward) + "," + str(round(ql.epsilon, 2)) + "," + "%d:%02d:%02d" % (h, m, s) + "\n")

            episode_step += 1

            if testMode == 'test':
                break

        env.close()
