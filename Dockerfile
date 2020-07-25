FROM tiryoh/ros-desktop-vnc:kinetic
MAINTAINER airkei

ENV USER ubuntu 
ENV HOME /home/${USER}
ENV SHELL /bin/bash

# Set locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:us
ENV LC_ALL en_US.UTF-8

# Default Shell Setup
# make /bin/sh symlink to bash instead of dash:
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash

# For buger_war scripts
RUN apt-get update
RUN apt-get install -y gnome-terminal at-spi2-core

# 1. Install ros kinetic
RUN apt-get install -y lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'; \
    apt-key adv --keyserver 'hkp://ha.pool.sks-keyservers.net:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654; \
    apt-get update; \
    apt-get install -y ros-kinetic-desktop-full

# For Gazebo8
RUN apt-get remove -y ros-kinetic-gazebo* libgazebo* gazebo*
RUN sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
RUN apt-get update
RUN apt-get install -y ros-kinetic-gazebo8-* ros-kinetic-turtlebot3-description

# Environment setting
RUN rosdep init; \
    rosdep update; \
    echo "source /opt/ros/kinetic/setup.bash" >> ${HOME}/.bashrc; \
    source ${HOME}/.bashrc

# Create workspace
RUN mkdir -p ~/catkin_ws/src; \
#    cd ~/catkin_ws/src; \
#    catkin_init_workspace; \
#    cd ~/catkin_ws/; \
#    catkin_make; \
    echo "source ${HOME}/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc; \
    source ${HOME}/.bashrc

# 2. Clone repo
#ARG REPO="https://github.com/OneNightROBOCON/burger_war"
RUN apt-get install git
#RUN cd ~/catkin_ws/src; \
#    git clone ${REPO} 

RUN echo "export GAZEBO_MODEL_PATH=$HOME/catkin_ws/src/burger_war/burger_war/models/" >> ${HOME}/.bashrc; \
    echo "export TURTLEBOT3_MODEL=burger" >> ${HOME}/.bashrc; \
    source ${HOME}/.bashrc

# 3. Install libraries
RUN apt-get install -y python-pip; \
    pip install requests flask; \
    apt-get install -y ros-kinetic-turtlebot3 ros-kinetic-turtlebot3-msgs ros-kinetic-turtlebot3-simulations; \
    apt-get install -y ros-kinetic-aruco-ros

# Additional Packages
# [MUST]DQN(Training)
RUN pip install numpy==1.16.6 \
                scipy==1.2.1 \
                gym==0.16.0 \
                Markdown==3.1.1 \
                setuptools==44.1.1 \
                grpcio==1.27.2 \
                mock==3.0.5 \
                gym==0.16.0 \
                tensorflow==1.14.0 \
                tensorflow-gpu==1.14.0 \
                keras==2.3.0 \
                flatten_json==0.1.7

# [MUST]OpenCV apps
RUN apt-get install -y ros-kinetic-opencv-apps

# [DEBUG]Joystick
RUN apt-get install -y ros-kinetic-joy  ros-kinetic-joystick-drivers

# [DEBUG]Jupyter
RUN pip install jupyter pandas

# Clean Cache
RUN apt-get clean; \
    rm -rf /var/lib/apt/lists/*
