#!usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import rospy
import random
import numpy as np
import os
from collections import deque
from std_msgs.msg import Float32MultiArray
from duelingQ_network import DuelingQNetwork
from torchsummary import summary
from setup import *
from losses import Regressions
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Paths for saving data
LOSS_DATA_DIR = os.path.dirname(os.path.realpath(__file__))
LOSS_DATA_DIR = LOSS_DATA_DIR.replace(
    'dueling_dqn_gazebo/nodes', 
    'dueling_dqn_gazebo/save_model/Dueling_DQN_trial_1/log_data'
)

class MemoryBuffer():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size

    def sample(self, count):
        count = min(count, len(self.buffer))
        batch = random.sample(self.buffer, count)
        
        states = np.float32([item[0] for item in batch])
        actions = np.float32([item[1] for item in batch])
        rewards = np.float32([item[2] for item in batch])
        next_states = np.float32([item[3] for item in batch])
        dones = np.bool8([item[4] for item in batch])
        
        return states, actions, rewards, next_states, dones

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

class DuelingQAgent():
    def __init__(self, state_size, action_size, mode, load_eps):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace(
            'dueling_dqn_gazebo/nodes', 
            'dueling_dqn_gazebo/save_model/Dueling_DQN_trial_1/pt_trial_1_'
        )
        self.result = Float32MultiArray()
        self.taus = torch.linspace(0.0, 1.0, 51, dtype=torch.float32)
        self.mode = mode
        self.load_model = mode == "test"
        self.load_episode = load_eps
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.995
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 64
        self.memory_size = replay_buffer_size
        self.RAM = MemoryBuffer(self.memory_size)

        self.Pred_model = DuelingQNetwork()
        self.Target_model = DuelingQNetwork()
        
        self.optimizer = optim.AdamW(self.Pred_model.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.training_loss = []

        if self.load_model:
            loaded_state_dict = torch.load(self.dirPath + str(self.load_episode) + '.pt')
            self.Pred_model.load_state_dict(loaded_state_dict)

    def updateTargetModel(self):
        self.Target_model.load_state_dict(self.Pred_model.state_dict())

    def getAction(self, state):
        if len(state.shape) == 2:
            state = torch.from_numpy(state).unsqueeze(0).view(1, 1, 144, 176)
            q_value = torch.mean(self.Pred_model(state), dim=2)

            if self.mode == "train" and np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            return int(torch.argmax(q_value))

    def quantile_huber_loss(self, target_quantiles, predicted_quantiles, kappa=0.5):
        errors = target_quantiles - predicted_quantiles
        errors_regression = Regressions.mean_square_error(predicted_quantiles, target_quantiles)
        tau_scaled = torch.abs(self.taus.unsqueeze(1).unsqueeze(1) - (errors.detach() < 0).float())
        quantile_loss = tau_scaled * errors_regression
        return quantile_loss.sum(dim=2).mean(dim=1).mean()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def TrainModel(self):
        states, actions, rewards, next_states, dones = self.RAM.sample(self.batch_size)
        
        states = torch.tensor(states).float()
        next_states = torch.tensor(next_states).float()
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        
        next_q_values = torch.max(self.Target_model(next_states), dim=1)[0].detach().numpy()
        q_values = rewards + self.discount_factor * next_q_values * (~dones)
        td_target = torch.tensor(q_values)

        predicted_values = self.Pred_model(states).gather(1, actions).squeeze()
        loss = self.quantile_huber_loss(td_target, predicted_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.Pred_model, self.Target_model)
