# ✨ Hướng dẫn Training
### Thuê Vast sử dụng Template 2.1.0
### Chạy dòng lệnh dưới đây trước
```
    sudo apt install dialog -y
```
### Chạy file update.bash bằng lệnh dưới dây để tiến hành cài đặt:
```
    cd RL_gazebo
```
```
    chmod +x update.bash
```
```
    ./update.bash
```
### Chạy file python3:
```
    python3 dueling_dqn_pt_model.py 
```
# ✨ Tăng tốc Training
### 1, Sử dụng `gzsever` ( Thay trong file launch)
- VD:
```
<node name="gazebo" pkg="gazebo_ros" type="gzserver" args="$(arg world)" output="screen"
```
### 2, Tăng tốc độ mô phỏng theo thời gian
- Tìm thẻ `physic` trong file `.wolrd`, tùy chỉnh các tham số:
- VD:
```
    <physics type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>100</real_time_factor>
      <real_time_update_rate>5000</real_time_update_rate>
    </physics>
```
