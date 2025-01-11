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
    def __init__(self, size, stack_size=4):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)  # Store the most recent frames for stacking
        
    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        states_array = np.float32([array[0] for array in batch])
        actions_array = np.float32([array[1] for array in batch])
        rewards_array = np.float32([array[2] for array in batch])
        next_states_array = np.float32([array[3] for array in batch])
        dones = np.bool8([array[4] for array in batch])
        
        return states_array, actions_array, rewards_array, next_states_array, dones
    
    def len(self):
        return self.len
    
    def add(self, s, a, r, new_s, d):
        # Nếu dữ liệu có nhiều kênh (ví dụ RGB), chuyển thành grayscale
        if len(s.shape) == 3:  # (C, H, W)
            s = np.mean(s, axis=0, keepdims=True)  # Giữ nguyên chiều H, W, nhưng chuyển thành 1 kênh
        if len(new_s.shape) == 3:
            new_s = np.mean(new_s, axis=0, keepdims=True)

        self.frames.append(s)
        if len(self.frames) == self.stack_size:
            stacked_state = np.stack(list(self.frames), axis=0)
        else:
            stacked_state = np.stack([self.frames[0]] * (self.stack_size - len(self.frames)) + list(self.frames), axis=0)

        self.frames.append(new_s)
        if len(self.frames) == self.stack_size:
            stacked_next_state = np.stack(list(self.frames), axis=0)
        else:
            stacked_next_state = np.stack([self.frames[0]] * (self.stack_size - len(self.frames)) + list(self.frames), axis=0)

        # Chọn frame cuối cùng (1 kênh) để đảm bảo tương thích
        stacked_state = stacked_state[-1:]  # Chọn frame cuối cùng (1 kênh)
        stacked_next_state = stacked_next_state[-1:]  # Chọn frame cuối cùng (1 kênh)

        # Lưu vào bộ nhớ
        transition = (stacked_state, a, r, stacked_next_state, d)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)



class DuelingQAgent():
    def __init__(self, state_size, action_size, mode, load_eps):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace(
            'dueling_dqn_gazebo/nodes', 
            'dueling_dqn_gazebo/save_model/Dueling_DQN_trial_1/pt_trial_1_'
        )
        self.result = Float32MultiArray()
        self.tau = 0.001
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
        if len(state.shape) == 2:  # Nếu trạng thái là ảnh 2D
            state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float()  # Thêm batch size và channel
        elif len(state.shape) == 3 and state.shape[0] > 1:  # Nếu trạng thái có nhiều kênh
            state = torch.mean(state, dim=0, keepdim=True).unsqueeze(0)  # Chuyển sang grayscale và thêm batch size
        
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

        # Chuyển đổi dữ liệu sang dạng tensor
        states = torch.tensor(states).float()
        next_states = torch.tensor(next_states).float()
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)  # (batch_size, 1)
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones).bool()

        # Tính next_q_values
        next_q_values = torch.max(self.Target_model(next_states), dim=2)[0].detach()

        # Mở rộng chiều của dones
        dones = dones.unsqueeze(-1)  # (batch_size, 1)
        dones = dones.expand_as(next_q_values)  # (batch_size, num_actions, num_quantiles)

        # Tính giá trị Q
        q_values = rewards.unsqueeze(-1) + self.discount_factor * next_q_values * (~dones)
        
        # Thay đổi cách tính td_target để có kích thước (batch_size, num_actions, num_quantiles)
        td_target = q_values.unsqueeze(2).expand_as(self.Pred_model(states))  # (batch_size, num_actions, num_quantiles)

        # Lấy predicted_values từ Pred_model
        predicted_values = self.Pred_model(states)  # (batch_size, num_actions, num_quantiles)

        # Kiểm tra và đảm bảo td_target và predicted_values có cùng kích thước
        assert td_target.size() == predicted_values.size(), \
            f"Size mismatch: td_target size {td_target.size()} vs predicted_values size {predicted_values.size()}"

        # Tính loss
        loss = self.loss_func(predicted_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.Pred_model, self.Target_model)










