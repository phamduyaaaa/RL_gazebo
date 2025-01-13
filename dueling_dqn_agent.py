#!usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
import rospy
import random
import numpy as np
import os
#from torchvision.transforms import functional as Fa
from losses import Regressions
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from duelingQ_network import DuelingQNetwork
from torchsummary import summary
LOSS_DATA_DIR = os.path.dirname(os.path.realpath(__file__))
LOSS_DATA_DIR = LOSS_DATA_DIR.replace('dueling_dqn_gazebo/nodes', 'dueling_dqn_gazebo/save_model/Dueling_DQN_trial_1/log_data')
from setup import *
class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        self.state_stack = deque(maxlen=4)  # To store 4 consecutive images

    def sample(self, count):
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        states_array = np.float32([array[0] for array in batch])
        actions_array = np.float32([array[1] for array in batch])
        rewards_array = np.float32([array[2] for array in batch])
        next_states_array = np.float32([array[3] for array in batch])
        dones = np.bool8([array[4] for array in batch])
        
        return states_array, actions_array, rewards_array, next_states_array, dones

    def add(self, s, a, r, new_s, d):
        # Append the new image to the state stack
        self.state_stack.append(s)
        
        # Ensure that the state stack has 4 images before adding to buffer
        if len(self.state_stack) == 4:
            stacked_state = np.stack(self.state_stack, axis=0)  # Shape: (4, height, width)
            transition = (stacked_state, a, r, new_s, d)
            self.len += 1 
            if self.len > self.maxSize:
                self.len = self.maxSize
            self.buffer.append(transition)

    def __len__(self):
        return self.len


class DuelingQAgent():
    def __init__(self, state_size, action_size, mode, load_eps):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('dueling_dqn_gazebo/nodes', 'dueling_dqn_gazebo/save_model/Dueling_DQN_trial_1/pt_trial_1_')
        self.result = Float32MultiArray()
        self.taus= torch.linspace(0.0, 1.0, 51, dtype=torch.float32)
        self.mode = mode
        self.load_model = False
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
        self.tau = 0.01
        self.batch_size = 64
        self.train_start = 64
        self.memory_size = replay_buffer_size
        self.RAM = MemoryBuffer(self.memory_size)
        self.num_actions = 5
        self.Pred_model = DuelingQNetwork()
        self.Target_model = DuelingQNetwork()
        
        self.optimizer = optim.AdamW(self.Pred_model.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.episode_loss = 0.0
        self.running_loss = 0.0
        self.training_loss = []
        self.x_episode = []
        self.counter = 0
        
        if self.mode == "test":
            self.load_model = True
        
        if self.load_model:
            loaded_state_dict = torch.load(self.dirPath + str(self.load_episode) + '.pt')
            self.Pred_model.load_state_dict(loaded_state_dict)

    def updateTargetModel(self):
        self.Target_model.load_state_dict(self.Pred_model.state_dict())
   
    def getAction(self, state):       
        # Kiểm tra và chuyển đổi state nếu cần
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            state = state.float()
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")
        
        # Không cần thêm batch dimension nữa vì state đã có shape phù hợp
        q_value = torch.mean(self.Pred_model(state), dim=2)  # Tính giá trị Q
        
        if self.mode == "train":
            if np.random.rand() <= self.epsilon:
                action = random.randrange(self.action_size)
                print(f"Random action selected: {action}")
            else:
                action = int(torch.argmax(q_value))
                print(f"Predicted action: {action}")
        elif self.mode == "test":
            action = int(torch.argmax(q_value))
            print(f"Predicted action (test mode): {action}")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        return action

    #FIXED
    def quantile_huber_loss(self, target_quantiles, predicted_quantiles):
        # Ensure target_quantiles has the correct shape: [batch_size, num_actions, num_quantiles, num_features]
        target_quantiles = target_quantiles.squeeze(1)  # Shape: [batch_size, num_quantiles, num_features]
        
        # Check if we need to add a new dimension (unsqueeze) to match num_actions
        target_quantiles = target_quantiles.unsqueeze(1)  # Shape: [batch_size, 1, num_quantiles, num_features]

        # Expand to match the actions dimension (expand to match [batch_size, num_actions, num_quantiles, num_features])
        target_quantiles = target_quantiles.expand(-1, self.num_actions, -1, -1)  # Shape: [batch_size, num_actions, num_quantiles, num_features]

        # Ensure predicted_quantiles has the correct number of dimensions: [batch_size, num_actions, num_quantiles, num_features]
        # Assuming predicted_quantiles has the shape [batch_size, num_actions], 
        # we need to expand it to have the same dimensions as target_quantiles.
        
        predicted_quantiles = predicted_quantiles.unsqueeze(2).unsqueeze(3)  # Add dimensions for num_quantiles and num_features
        predicted_quantiles = predicted_quantiles.expand(-1, -1, 5, 51)  # Expand to match the quantiles and features

        # Now, both tensors should have the same shape: [batch_size, num_actions, num_quantiles, num_features]
        assert target_quantiles.shape == predicted_quantiles.shape, \
            f"Shape mismatch: target_quantiles {target_quantiles.shape}, predicted_quantiles {predicted_quantiles.shape}"

        # Calculate the error between target and predicted quantiles
        errors = target_quantiles - predicted_quantiles  # Now both tensors should have the same shape

        # Apply Huber loss function
        huber_loss = torch.where(errors.abs() < 1.0, 0.5 * errors ** 2, errors.abs() - 0.5)
        
        # Calculate and return the final loss
        loss = huber_loss.mean()
        return loss





    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(1e-2 * local_param.data + (1.0 - 1e-2) * target_param.data)
    #FIXED
    def TrainModel(self):
        states, actions, rewards, next_states, dones = self.RAM.sample(self.batch_size)

        
        states = np.array(states).squeeze()  # Shape: (batch_size, 4, height, width)
        next_states = np.array(next_states).squeeze()  # Shape: (batch_size, 176, 144)
        
        # Chuyển thành tensor PyTorch
        states = torch.tensor(states).float()  # (batch_size, 4, height, width)
        next_states = torch.tensor(next_states).float()

        # Thêm chiều kênh vào next_states (chuyển từ [batch_size, height, width] thành [batch_size, 4, height, width])
        next_states = next_states.unsqueeze(1).expand(-1, 4, -1, -1)

        actions = torch.Tensor(actions).type(torch.int64).unsqueeze(-1)

        next_q_value = torch.max(self.Target_model(next_states), dim=1)[0].detach().numpy()

        q_value = torch.zeros(self.batch_size, 51, dtype=torch.float32)

        for i in range(self.batch_size):
            if dones[i]:
                q_value[i] = torch.tensor(rewards[i], dtype=torch.float32)
            else:
                q_value[i] = rewards[i] + self.discount_factor * torch.tensor(next_q_value[i], dtype=torch.float32)


        td_target = q_value.unsqueeze(1).expand(-1, self.num_actions, -1)  # Make sure q_value is of the same shape

        predicted_values = self.Pred_model(states).gather(1, actions.view(self.batch_size, 1, 1).expand(self.batch_size, 1, 5)).squeeze()


        self.loss = self.quantile_huber_loss(td_target, predicted_values)



        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.soft_update(self.Pred_model, self.Target_model)
        self.episode_loss += predicted_values.shape[0] * self.loss.item()
        self.running_loss += self.loss.item()
        cal_loss = self.episode_loss / len(states)
        self.training_loss.append(cal_loss)
        self.counter += 1
        self.x_episode.append(self.counter)
        np.savetxt(LOSS_DATA_DIR + '/loss.csv', self.training_loss, delimiter=' , ')



