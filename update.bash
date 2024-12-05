#!/bin/bash
  #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_
# Des: Sửa các lỗi nêu ra trong README           #
# Mục đích: Tự động cài đặt, tải và sửa phần mềm #
# Nếu chưa cài dialog, hãy chạy dòng lệnh dưới:  #
# sudo apt install dialog -y                     #
# Author: phamduyaaaa                            #
  #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
(
    echo "0";"Pha 1 cốc cà phê trong đi, vì bạn là cậu bé thư giãn ☕ "
    echo "5"; echo "Đang kiểm tra lại hệ thống..."
    sudo apt update -y
    sleep 1

    echo "20"; echo "Cấp quyền truy cập cho /home/kasm-user..."
    sudo chown -R kasm-user:kasm-user /home/kasm-user
    sleep 1

    echo "40"; echo "Cài đặt Pytorch, TorchSummary..."
    sudo pip install torch
    sudo pip install torchsummary
    sleep 1

    echo "60"; echo "Tải Models..."
    git clone https://github.com/nguyendatxtnd/models.git
    sudo chown -R $(whoami):$(whoami) models
    sleep 1

    echo "70"; echo "Tải mapRL"
    git clone https://github.com/nguyendatxtnd/mapRL
    echo "75"; echo "Tiến hành catkin_make"
    cd mapRL
    catkin_make

    echo "80"; echo "Tải file dueling_dqn_gazebo"
    git clone https://github.com/phamduyaaaa/RL_gazebo.git
    sudo chown -R $(whoami):$(whoami) RL_gazebo
    sleep 1
    echo "90"; echo "Cấp quyền ROOT"
    sudo chown -R $(whoami):$(whoami) ../mapRL
    cd ..
    sleep 1

    echo "100"; echo "Hoàn tất!"
) | dialog --gauge "Đang xử lý..." 10 70 0
