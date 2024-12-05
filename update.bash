#!/bin/bash

# Des: Sửa các lỗi nêu ra trong README
# Mục đích: Tự động cài đặt, tải và sửa phần mềm
# Author: phamduyaaaa
echo "Tiến hành sửa các lỗi nêu ra trong README"
echo "Đang kiểm tra lại hệ thống"
sudo apt update -y
echo "Đợi tý đang sửa 😅"
cd /home/$(whoami)
echo "👉 0. Cấp quyền truy cập cho /home/kasm-user"
sudo chown -R kasm-user:kasm-user /home/kasm-user
echo "👉 1. Cài Pytorch, TorchSummary"
sudo pip install torch
sudo pip install torchsummary
echo "👉 2. Bổ sung Models"
git clone https://github.com/nguyendatxtnd/models.git
sudo chown -R $(whoami):$(whoami) models
echo "👉 3. Sửa H x W "
git clone https://github.com/nguyendatxtnd/mapRL
su mapRL
catkin_make
cdo chown -R $(whoami):$(whoami) mapRL
cdd /home/$(whoami)
echo "👉 4. Sửa file dueling_dqn_gazebo"
cd dueling_dqn_gazebo
rm dueling_dqn_pt_model.py
rm dueling_dqn_agent.py
rm duelingQ_network.py
git clone https://github.com/phamduyaaaa/RL_gazebo.git
sudo chown -R $(whoami):$(whoami) RL_gazebo











