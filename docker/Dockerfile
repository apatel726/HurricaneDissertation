FROM tensorflow/tensorflow:latest-gpu-jupyter
COPY files/ /tf/huraim

RUN git clone --single-branch --branch master https://github.com/apatel726/HurricaneDissertation.git
RUN git config --global user.email "vm"
RUN git config --global user.name "vm"

# Install requirements for cartopy
RUN apt install -y libproj-dev proj-data proj-bin libgeos-dev python3-dev

# Python requirements
RUN pip install -r /tf/huraim/requirements.txt

# This is copied from the base image reference here to allow a password
# https://hub.docker.com/layers/tensorflow/tensorflow/1.14.0-py3-jupyter/images/sha256-07321f6227cdb5c60e9276a26f5e8d3168237ad032203275ed27357289c4c6f4?context=explore
CMD "bash" "-c" "source /etc/bash.bashrc && SHELL=bash jupyter lab --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token=huraim1337 && SHELL=bash tensorboard --logdir /tf/HurricaneDissertation/hurricane_ai/models/ --bind_all"
