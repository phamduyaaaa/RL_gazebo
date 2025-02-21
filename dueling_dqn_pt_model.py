#!usr/bin/env python3

import torch
import rospy
import numpy as np
import os
import time 
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from std_msgs.msg import Float32MultiArray
from src.dueling_dqn_env import Env
from dueling_dqn_agent import *
from setup import *

dirPath = os.path.dirname(os.path.realpath(__file__))
LOG_DATA_DIR = dirPath + '/log_data'
EPISODES = num_episodes    

def Training():
    mode = "train"
    load_episodes = 0
    rospy.init_node('dueling_dqn_gazebo_pt_model')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    # state_size = 182
    state_size = 1
    action_size = 5
    
    # define the environment
    env = Env(action_size)

    # define the agent
    agent = DuelingQAgent(state_size, action_size, mode, load_episodes)
    episodes = []
    steps_per_episode = []
    total_rewards = []
    global_step = 0
    goal_counters_array = []
    
    # Init log files
    log_sim_info = open(LOG_DATA_DIR+'/LogInfo.txt','w+')
    
    # Date / Time
    start_time = time.time()
    now_start = datetime.now()
    dt_string_start = now_start.strftime("%d/%m/%Y %H:%M:%S")

    # Log date to files
    text = '\r\n' + '****************************************************************\n'
    text = text + 'SIMULATION START ==> ' + dt_string_start + '\r\n'
    text = text + 'INITIAL ROBOT POSITION = ( %.2f , %.2f , %.2f ) \r\n' % (env.init_x, env.init_y, 0.0)
    text = text + '****************************************************************\n'
    print(text)
    log_sim_info.write(text)
    position_log_file = open(LOG_DATA_DIR + '/robot_positions.csv', 'w', newline='') #NEWx3
    position_writer = csv.writer(position_log_file)
    position_writer.writerow(["Episode", "Step", "X", "Y", "Theta"])  # Ghi header

    for e in range(agent.load_episode + 1, EPISODES):
        text = '\r\n' + '_____ EPISODE: ' + str(e) + ' _____' + '\r\n'
        text = text + '----------------------------------------------------------------\n'
        print(text)
        done = False
        counters = 0
        state = env.reset()
        reward_per_episode = 0
        if len(state.shape)==2:
            for step in range(agent.episode_step):
                action = agent.getAction(state)
                next_state, reward, done, counters = env.step(action)
                x, y, theta = env.robot_position()  # Giả sử hàm này trả về vị trí hiện tại của robot
                position_writer.writerow([e, step, x, y, theta]) # NEWx2 
                done = np.bool8(done)
                
                agent.RAM.add(state, action, reward, next_state, done)
                if agent.RAM.len >= agent.train_start:
                    if global_step % 1 == 0:
                        agent.TrainModel()
                reward_per_episode += reward
                state = next_state
                get_action.data = [action, reward_per_episode, reward]
                pub_get_action.publish(get_action)

                if e == 1500 or e == 1000:
                    torch.save(agent.Pred_model.state_dict(), agent.dirPath + str(e) + '.pt')

                if step >= 1000:
                    print('\n==> Time out! Maxed step per episode\n')
                    done = True

                if done:
                    result.data = [reward_per_episode, action]
                    pub_result.publish(result)
                    agent.updateTargetModel()
                    total_rewards.append(reward_per_episode)
                    steps_per_episode.append(step)
                    goal_counters_array.append(counters)
                    episodes.append(e)
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)

                    rospy.loginfo('Episode: %d , Reward per episode: %.2f , Memory: %d , Epsilon: %.4f , Time: %d:%02d:%02d',
                                    e, reward_per_episode, agent.RAM.len, agent.epsilon, h, m, s)
                    text = text + 'Episode: %d , Reward per episode: %.2f , Memory: %d , Epsilon: %.4f , Time: %d:%02d:%02d \r\n'%\
                        (e, reward_per_episode, agent.RAM.len, agent.epsilon, h, m, s)
                    text = text + '----------------------------------------------------------------\r\n'
                    log_sim_info.write('\r\n'+text)
                    
                    param_keys = ['epsilon']
                    param_values = [agent.epsilon]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                global_step += 1    # step counting when the robot takes action for each iteration
                if global_step % agent.target_update == 0:
                    rospy.loginfo("UPDATE TARGET NETWORK")
                    agent.updateTargetModel()
                

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Save data to directory
        np.savetxt(LOG_DATA_DIR + '/steps_per_episode.csv', steps_per_episode, delimiter = ' , ')
        np.savetxt(LOG_DATA_DIR + '/reward_per_episode.csv', total_rewards, delimiter = ' , ')
        np.savetxt(LOG_DATA_DIR + '/goal_counters_per_episode.csv', goal_counters_array, delimiter = ' , ')
    
    # Close the log file
    log_sim_info.close()
    position_log_file.close() # Newx1
        
        
if __name__ == '__main__':
    try:
        Training()
            
    except rospy.ROSInterruptException:
        print("<--------- Training completed --------->")
        print('Simulation terminated!')
        pass
