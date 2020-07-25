FROM tiryoh/ros-desktop-vnc:kinetic
MAINTAINER airkei

ENV USER ubuntu 
ENV HOME /home/${USER}
ENV SHELL /bin/bash


# Default Shell Setup
# make /bin/sh symlink to bash instead of dash:
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash

# Install lsb-release
RUN apt-get update
RUN apt-get install -y lsb-release

# 1. Install ros kinetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'; \
    apt-key adv --keyserver 'hkp://ha.pool.sks-keyservers.net:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654; \
    apt-get update; \
    apt-get install -y ros-kinetic-desktop-full

# Environment setting
RUN rosdep init; \
    rosdep update; \
    echo "source /opt/ros/kinetic/setup.bash" >> ${HOME}/.bashrc; \
    source ${HOME}/.bashrc

# Create workspace
RUN mkdir -p ~/catkin_ws/src; \
    cd ~/catkin_ws/src; \
    catkin_init_workspace; \
    cd ~/catkin_ws/; \
    catkin_make; \
    echo "source ${HOME}/catkin_ws/devel/setup.bash" >> ${HOME}/.bashrc; \
    source ${HOME}/.bashrc

# 2. Clone repo
ARG REPO="https://github.com/OneNightROBOCON/burger_war"
RUN apt-get install git
RUN cd ~/catkin_ws/src; \
    git clone ${REPO} 

RUN echo "export GAZEBO_MODEL_PATH=$HOME/catkin_ws/src/burger_war/burger_war/models/" >> ${HOME}/.bashrc; \
    echo "export TURTLEBOT3_MODEL=burger" >> ${HOME}/.bashrc; \
    source ${HOME}/.bashrc

# 3. Install libraries
RUN apt-get install -y python-pip; \
    pip install requests flask; \
    apt-get install -y ros-kinetic-turtlebot3 ros-kinetic-turtlebot3-msgs ros-kinetic-turtlebot3-simulations; \
    apt-get install -y ros-kinetic-aruco-ros

# Clean Cache
RUN apt-get clean; \
    rm -rf /var/lib/apt/lists/*

