#!/bin/bash
  #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_
# Des: Sửa các lỗi trong Dockerfile 2.1.0        #
# Mục đích: Tự động cài đặt, tải và sửa phần mềm #
# Nếu chưa cài dialog, hãy chạy dòng lệnh dưới:  #
# sudo apt install dialog -y                     #
# Author: phamduyaaaa                            #
  #_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#

# Bắt đầu tiến trình với dialog
(
    # Tiến độ bắt đầu từ 0%
    echo "0"; echo "Author: phamduyaaaa "
    sleep 1

    echo "10"; echo "Đang kiểm tra lại hệ thống..."
    sudo apt update -y
    sleep 1

    echo "20"; echo "Cấp quyền truy cập cho /home/kasm-user..."
    sudo chown -R $(whoami):$(whoami) /home/$(whoami)
    sleep 1

    echo "30"; echo "Cài đặt Pytorch, TorchSummary..."
    sudo pip install torch
    sudo pip install torchsummary
    sleep 1

    echo "40"; echo "Tải Models..."
    cd /home/$(whoami)
    git clone https://github.com/nguyendatxtnd/models.git
    sudo chown -R $(whoami):$(whoami) models
    sleep 1

    echo "50"; echo "Tải file dueling_dqn_gazebo"
    git clone https://github.com/phamduyaaaa/RL_gazebo.git
    sudo chown -R $(whoami):$(whoami) RL_gazebo
    sleep 1

    echo "60"; echo "Tải mapRL"
    git clone https://github.com/nguyendatxtnd/mapRL
    sudo chown -R $(whoami):$(whoami) ../mapRL
    sleep 1

    echo "70"; echo "Tiến hành catkin_make"
    cd mapRL
    catkin_make
    source devel/setup.bash
    sleep 1

    echo "80"; echo "Cập nhật file .bashrc"
    BASHRC_FILE="$HOME/.bashrc"
    MAPRL_SETUP="$HOME/mapRL/devel/setup.bash"
    echo "source $MAPRL_SETUP" >> "$BASHRC_FILE"
    sleep 1

    echo "90"; echo "Cập nhật xong, sẵn sàng sử dụng!"
    sleep 1

    echo "100"; echo "Hoàn tất!"
) | dialog --gauge "Pha 1 cốc cà phê trong đi, vì bạn là cậu bé thư giãn ☕ ..." 10 40 0
