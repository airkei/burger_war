#!/usr/bin/env python

import gym
from gym import wrappers
import time
import datetime
from distutils.dir_util import copy_tree
import os
import json
import rospy

import deepq

class Train:
    def detect_monitor_files(self, training_dir):
        return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

    def clear_monitor_files(self, training_dir):
        files = self.detect_monitor_files(training_dir)
        if len(files) == 0:
            return
        for file in files:
            print(file)
            os.unlink(file)

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
        weights_path = resume_path + '.h5'
        monitor_path = resume_path
        params_json  = resume_path + '.json'

        if resume_epoch == "0":
            #Each time we take a sample and update our weights it is called a mini-batch.
            #Each time we run through the entire dataset, it's called an epoch.
            #PARAMETER LIST
            save_interval = rospy.get_param("/burger/save_interval")
            epochs = rospy.get_param("/burger/epochs")
            steps = rospy.get_param("/burger/steps")
            explorationRate = rospy.get_param("/burger/explorationRate")
            minibatch_size = rospy.get_param("/burger/minibatch_size")
            learningRate = rospy.get_param("/burger/learningRate")
            discountFactor = rospy.get_param("/burger/discountFactor")
            memorySize = rospy.get_param("/burger/memorySize")
            updateTargetNetwork = rospy.get_param("/burger/updateTargetNetwork")
            learnStart = rospy.get_param("/burger/learnStart")
            network_inputs = rospy.get_param("/burger/network_inputs")
            network_outputs = rospy.get_param("/burger/network_outputs")
            network_structure = rospy.get_param("/burger/network_structure")
            current_epoch = 0

            epsilon_decay = rospy.get_param("/burger/epsilon_decay")

            vel_max_x = rospy.get_param("/burger/vel_max_x")
            vel_min_x = rospy.get_param("/burger/vel_min_x")
            vel_max_z = rospy.get_param("/burger/vel_max_z")

            deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
            deepQ.initNetworks(network_structure)
        else:
            #Load weights, monitor info and parameter info.
            #ADD TRY CATCH fro this else
            with open(params_json) as outfile:
                d = json.load(outfile)
                save_interval = d.get('save_interval')
                epochs = d.get('epochs')
                steps = d.get('steps')
                if testMode == 'test':
                    explorationRate = 0
                else:
                    explorationRate = d.get('explorationRate')
                minibatch_size = d.get('minibatch_size')
                learnStart = d.get('learnStart')
                discountFactor = d.get('discountFactor')
                memorySize = d.get('memorySize')
                network_outputs = d.get('network_outputs')
                updateTargetNetwork = d.get('updateTargetNetwork')
                learningRate = d.get('learningRate')
                network_inputs = d.get('network_inputs')
                network_outputs = d.get('network_outputs')
                network_structure = d.get('network_structure')
                current_epoch = d.get('current_epoch')

                epsilon_decay = d.get('epsilon_decay')

                vel_max_x = d.get('vel_max_x')
                vel_min_x = d.get('vel_min_x')
                vel_max_z = d.get('vel_max_z')

            deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
            deepQ.initNetworks(network_structure)

            deepQ.loadWeights(weights_path)

            self.clear_monitor_files(outdir)
            if not os.path.exists(monitor_path):
                os.makedirs(monitor_path)
            copy_tree(monitor_path,outdir)

        env._max_episode_steps = steps # env returns done after _max_episode_steps
        prod = False
        if testMode == 'test':
            prod = True
        env = gym.wrappers.Monitor(env, outdir, force=not prod, resume=prod)

        lastScores = [0] * save_interval
        lastScoresIndex = 0
        lastFilled = False
        stepCounter = 0

        start_time = time.time()

        #start iterating from 'current epoch'.
        env.set_mode(testMode, caMode, network_outputs, vel_max_x, vel_min_x, vel_max_z)
        for epoch in xrange(current_epoch+1, epochs+1, 1):
            observation = env.reset()

            cumulated_reward = 0
            done = False
            episode_step = 0

            # run until env returns done
            while not done:
                # env.render()
                qValues = deepQ.getQValues(observation)

                action = deepQ.selectAction(qValues, explorationRate)

                newObservation, reward, done, info = env.step(action)
                cumulated_reward += reward

                deepQ.addMemory(observation, action, reward, newObservation, done)

                if ((testMode != 'test') and (stepCounter >= learnStart)):
                    if stepCounter <= updateTargetNetwork:
                        deepQ.learnOnMiniBatch(minibatch_size, False)
                    else :
                        deepQ.learnOnMiniBatch(minibatch_size, True)

                observation = newObservation

                if done:
                    lastScores[lastScoresIndex] = episode_step
                    lastScoresIndex += 1
                    if lastScoresIndex >= save_interval:
                        lastFilled = True
                        lastScoresIndex = 0
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    if not lastFilled:
                        print ("EP " + str(epoch) + " - " + format(episode_step + 1) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))
                    else :
                        print ("EP " + str(epoch) + " - " + format(episode_step + 1) + " Episode steps - last Steps : " + str((sum(lastScores) / len(lastScores))) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))

                        if (epoch % save_interval) == 0:
                            #save model weights and monitoring data every save_interval epochs.
                            deepQ.saveModel(path+str(epoch)+'.h5')
                            env._flush()
                            copy_tree(outdir,path+str(epoch))
                            #save simulation parameters.
                            parameter_keys = ['save_interval', 'epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch',epsilon_decay,'vel_max_x','vel_min_x','vel_max_z']
                            parameter_values = [save_interval, epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch, epsilon_decay, vel_max_x, vel_min_x, vel_max_z]
                            parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                            with open(path+str(epoch)+'.json', 'w') as outfile:
                                json.dump(parameter_dictionary, outfile)

                    with open(resultpath, mode='a') as f:
                        f.write(str(epoch) + "," + format(episode_step + 1) + "," + str(cumulated_reward) + "," + str(round(explorationRate, 2)) + "," + "%d:%02d:%02d" % (h, m, s) + "\n")

                stepCounter += 1
                if stepCounter % updateTargetNetwork == 0:
                    deepQ.updateTargetNetwork()
                    print ("updating target network")

                episode_step += 1

            if testMode == 'train':
                explorationRate *= epsilon_decay #epsilon decay
                # explorationRate -= (2.0/epochs)
                explorationRate = max (0.05, explorationRate)

            if testMode == 'test':
                break

        env.close()
