#!/bin/bash

echo "#git install"
sudo apt-get install -y git

echo "#forkしたgitをダウンロード(環境に合わせて修正する事)"
cd ~/catkin_ws/src
git clone https://github.com/airkei/burger_war.git

echo "model path add .bashrc"
echo "export GAZEBO_MODEL_PATH=$HOME/catkin_ws/src/burger_war/burger_war/models/" >> ~/.bashrc
source ~/.bashrc

echo "# pip のインストール"
sudo apt-get install -y python-pip

echo "#　requests flask のインストール"
sudo pip install requests flask

echo "# turtlebot3 ロボットモデルのインストール"
sudo apt-get install -y ros-kinetic-turtlebot3 ros-kinetic-turtlebot3-msgs ros-kinetic-turtlebot3-simulations

echo "# aruco (ARマーカー読み取りライブラリ）"
sudo apt-get install -y ros-kinetic-aruco-ros

echo "# catkin_make"
cd ~/catkin_ws
catkin_make


echo "# python 2.7 libraries"
pip install numpy==1.16.6 \
                scipy==1.2.1 \
                gym==0.16.0 \
                Markdown==3.1.1 \
                setuptools==44.1.1 \
                grpcio==1.27.2 \
                mock==3.0.5 \
                tensorflow==1.14.0 \
                tensorflow-gpu==1.14.0 \
                keras==2.3.0 \
                flatten_json==0.1.7

echo "# ros-kinetic-opencv-apps for circle detection"
sudo apt-get install -y ros-kinetic-opencv-apps
