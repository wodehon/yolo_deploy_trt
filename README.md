source /opt/ros/noetic/setup.zsh

mkdir -p catkin_ws/src && cd catkin_ws

catkin_init_workspace

cd src

git clone

cd ..

catkin_make


Notice:
1. build cv_bridge for opencv4.5.5, sudo make install

2. .pt -> .engine from yolov5-v7.0
